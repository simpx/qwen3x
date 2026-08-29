// main.cpp -- token-line server and client over a Unix socket.
//
// One connection carries one request and one response. Every line ends in LF;
// an empty line separates parameters from tokens, and another ends the tokens.
// A token line may contain one or more decimal token IDs separated by spaces.
//
// Request:
//   POST /infer
//   max-tokens: 64
//   temperature: 0
//
//   123 456
//   789
//
// Response:
//   200 OK
//   cached-tokens: 2
//
//   9707 198
//   13
//
// The final empty line means normal completion. EOF before it means failure or
// cancellation. See protocol.md for parameters, errors and exact semantics.

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "qwen35.h"

namespace {

constexpr size_t MAX_LINE = 4 * 1024 * 1024;
constexpr size_t ERROR_SIZE = 512;

struct Fd {
    int value = -1;
    explicit Fd(int value = -1) : value(value) {}
    ~Fd() { if (value >= 0) ::close(value); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : value(other.value) { other.value = -1; }
};

class Reader {
public:
    explicit Reader(int fd) : fd_(fd) {}

    bool line(std::string& output) {
        while (true) {
            const size_t end = buffer_.find('\n');
            if (end != std::string::npos) {
                output = buffer_.substr(0, end);
                buffer_.erase(0, end + 1);
                return output.find('\r') == std::string::npos;
            }
            if (buffer_.size() >= MAX_LINE) return false;
            char data[4096];
            const ssize_t count = ::recv(fd_, data, sizeof(data), 0);
            if (count == 0) return false;
            if (count < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            buffer_.append(data, static_cast<size_t>(count));
        }
    }

private:
    int fd_;
    std::string buffer_;
};

struct Request {
    std::vector<int> prompt;
    int max_tokens = 0;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float presence_penalty = 0.0f;
    uint64_t seed = 0;
};

struct ServerOptions {
    std::string socket_path;
    std::string model_path;
    int parallel = 1;
    int context = 4096;
    bool mock = false;
};

struct ClientOptions {
    std::string socket_path;
    int max_tokens = 128;
};

template <typename Integer>
bool integer(const std::string& text, Integer& output) {
    if (text.empty()) return false;
    Integer value{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end) return false;
    output = value;
    return true;
}

bool decimal(const std::string& text, float& output) {
    if (text.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() ||
        !std::isfinite(value)) return false;
    output = value;
    return true;
}

bool send_all(int fd, const std::string& text) {
    size_t sent = 0;
    while (sent < text.size()) {
        const ssize_t count = ::send(
            fd, text.data() + sent, text.size() - sent, MSG_NOSIGNAL
        );
        if (count < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (count == 0) return false;
        sent += static_cast<size_t>(count);
    }
    return true;
}

const char* reason(int status) {
    switch (status) {
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Context Too Large";
        default: return "Internal Error";
    }
}

void send_error(int fd, int status, std::string message) {
    for (char& character : message) {
        if (character == '\n' || character == '\r') character = ' ';
    }
    send_all(fd, std::to_string(status) + " " + reason(status) +
                 "\nmessage: " + message + "\n\n\n");
}

bool read_tokens(Reader& reader, std::vector<int>& tokens,
                 int limit, int& status, std::string& error) {
    std::string line;
    while (reader.line(line)) {
        if (line.empty()) return true;
        if (line.front() == ' ' || line.back() == ' ' ||
            line.find('\t') != std::string::npos) {
            error = "invalid token line";
            return false;
        }
        std::istringstream words(line);
        std::string word;
        while (words >> word) {
            int token = -1;
            if (!integer(word, token) || token < 0 || token >= q35_vocab_size()) {
                error = "invalid token ID";
                return false;
            }
            tokens.push_back(token);
            if (static_cast<int>(tokens.size()) > limit) {
                status = 413;
                error = "prompt exceeds context size";
                return false;
            }
        }
    }
    error = "incomplete token section";
    return false;
}

bool read_request(int fd, int context, Request& request,
                  int& status, std::string& error) {
    Reader reader(fd);
    std::string line, method, path, extra;
    if (!reader.line(line)) {
        error = "incomplete request line";
        return false;
    }
    std::istringstream first(line);
    if (!(first >> method >> path) || (first >> extra)) {
        error = "invalid request line";
        return false;
    }
    if (method != "POST") {
        status = 405;
        error = "only POST is supported";
        return false;
    }
    if (path != "/infer") {
        status = 404;
        error = "endpoint not found";
        return false;
    }

    unsigned seen = 0;
    while (reader.line(line) && !line.empty()) {
        const size_t split = line.find(": ");
        if (split == std::string::npos || split == 0 || split + 2 == line.size()) {
            error = "invalid parameter line";
            return false;
        }
        const std::string name = line.substr(0, split);
        const std::string value = line.substr(split + 2);
        unsigned bit = 0;
        bool valid = false;
        if (name == "max-tokens") {
            bit = 1u << 0;
            valid = integer(value, request.max_tokens) && request.max_tokens > 0;
        } else if (name == "temperature") {
            bit = 1u << 1;
            valid = decimal(value, request.temperature) &&
                    request.temperature >= 0.0f && request.temperature <= 2.0f;
        } else if (name == "top-k") {
            bit = 1u << 2;
            valid = integer(value, request.top_k) && request.top_k >= 0;
        } else if (name == "top-p") {
            bit = 1u << 3;
            valid = decimal(value, request.top_p) &&
                    request.top_p > 0.0f && request.top_p <= 1.0f;
        } else if (name == "presence-penalty") {
            bit = 1u << 4;
            valid = decimal(value, request.presence_penalty) &&
                    request.presence_penalty >= -2.0f &&
                    request.presence_penalty <= 2.0f;
        } else if (name == "seed") {
            bit = 1u << 5;
            valid = integer(value, request.seed);
        } else {
            error = "unknown parameter: " + name;
            return false;
        }
        if (seen & bit) {
            error = "duplicate parameter: " + name;
            return false;
        }
        if (!valid) {
            error = "invalid parameter: " + name;
            return false;
        }
        seen |= bit;
    }
    if (line.empty() && !(seen & 1u)) {
        error = "max-tokens is required";
        return false;
    }
    if (!line.empty()) {
        error = "incomplete parameter section";
        return false;
    }
    if (!read_tokens(reader, request.prompt, context, status, error)) return false;
    if (request.prompt.empty()) {
        error = "prompt must contain a token ID";
        return false;
    }
    if (request.prompt.size() + static_cast<size_t>(request.max_tokens) >
        static_cast<size_t>(context)) {
        status = 413;
        error = "prompt and completion exceed context size";
        return false;
    }
    return true;
}

bool peer_closed(int fd, int timeout_ms) {
    pollfd event{fd, POLLIN, 0};
#ifdef POLLRDHUP
    event.events |= POLLRDHUP;
#endif
    const int result = ::poll(&event, 1, timeout_ms);
    if (result < 0) return errno != EINTR;
    if (result == 0) return false;
    if (event.revents & (POLLHUP | POLLERR | POLLNVAL)) return true;
#ifdef POLLRDHUP
    if (event.revents & POLLRDHUP) return true;
#endif
    if (event.revents & POLLIN) {
        char byte;
        return ::recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
    }
    return false;
}

q35_session* acquire(int fd, q35_session_manager* manager,
                     const std::vector<int>& prompt) {
    while (true) {
        q35_session* session = nullptr;
        char error[ERROR_SIZE]{};
        const int result = q35_session_manager_acquire(
            manager, prompt.data(), static_cast<int>(prompt.size()),
            &session, error, sizeof(error)
        );
        if (result == Q35_OK) return session;
        if (result != Q35_BUSY) {
            send_error(fd, 500, error[0] ? error : "cannot acquire Session");
            return nullptr;
        }
        if (peer_closed(fd, 10)) return nullptr;
    }
}

struct SessionLease {
    q35_session_manager* manager;
    q35_session* session;
    bool keep = false;
    ~SessionLease() { q35_session_manager_release(manager, session, keep); }
};

void handle(int fd, q35_session_manager* manager, int context) {
    Fd connection(fd);
    Request request;
    int status = 400;
    std::string error;
    if (!read_request(fd, context, request, status, error)) {
        send_error(fd, status, error);
        return;
    }

    q35_session* session = acquire(fd, manager, request.prompt);
    if (!session) return;
    SessionLease lease{manager, session};

    char native_error[ERROR_SIZE]{};
    int cached = 0;
    if (q35_session_sync(
            session, request.prompt.data(), static_cast<int>(request.prompt.size()),
            static_cast<int>(request.prompt.size()), &cached,
            native_error, sizeof(native_error)) != Q35_OK) {
        send_error(fd, 500, native_error);
        return;
    }
    if (!send_all(fd, "200 OK\ncached-tokens: " +
                       std::to_string(cached) + "\n\n")) return;

    std::vector<int> generated;
    uint64_t rng = request.seed;
    for (int index = 0; index < request.max_tokens; ++index) {
        const int token = request.temperature == 0.0f &&
                          request.presence_penalty == 0.0f
            ? q35_session_argmax(session)
            : q35_session_sample(
                session, request.temperature, request.top_k, request.top_p,
                request.presence_penalty,
                generated.empty() ? nullptr : generated.data(),
                static_cast<int>(generated.size()), &rng
            );
        if (token < 0) return;
        if (q35_token_is_stop(token)) break;
        if (!send_all(fd, std::to_string(token) + "\n")) return;
        generated.push_back(token);
        if (index + 1 == request.max_tokens) break;
        if (q35_session_eval(session, token,
                             native_error, sizeof(native_error)) != Q35_OK) return;
    }
    if (!send_all(fd, "\n")) return;
    lease.keep = true;
}

int unix_socket(const std::string& path, std::string& error) {
    sockaddr_un address{};
    if (path.empty() || path.size() >= sizeof(address.sun_path)) {
        error = "Unix socket path is empty or too long";
        return -1;
    }
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        error = std::string("socket: ") + std::strerror(errno);
        return -1;
    }
    return fd;
}

sockaddr_un unix_address(const std::string& path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    return address;
}

int run_server(const ServerOptions& options, std::string& error) {
    char native_error[ERROR_SIZE]{};
    q35_engine* raw_engine = nullptr;
    q35_engine_options engine_options{options.model_path.c_str(), options.mock};
    if (q35_engine_create(&engine_options, &raw_engine,
                          native_error, sizeof(native_error)) != Q35_OK) {
        error = native_error;
        return 1;
    }
    std::unique_ptr<q35_engine, decltype(&q35_engine_destroy)>
        engine(raw_engine, q35_engine_destroy);

    q35_session_manager* raw_manager = nullptr;
    if (q35_session_manager_create(
            engine.get(), options.parallel, options.context, &raw_manager,
            native_error, sizeof(native_error)) != Q35_OK) {
        error = native_error;
        return 1;
    }
    std::unique_ptr<q35_session_manager, decltype(&q35_session_manager_destroy)>
        manager(raw_manager, q35_session_manager_destroy);

    struct stat existing{};
    if (::lstat(options.socket_path.c_str(), &existing) == 0) {
        if (!S_ISSOCK(existing.st_mode)) {
            error = "socket path exists and is not a socket";
            return 1;
        }
        if (::unlink(options.socket_path.c_str()) != 0) {
            error = std::string("unlink: ") + std::strerror(errno);
            return 1;
        }
    } else if (errno != ENOENT) {
        error = std::string("lstat: ") + std::strerror(errno);
        return 1;
    }

    Fd listener(unix_socket(options.socket_path, error));
    if (listener.value < 0) return 1;
    const sockaddr_un address = unix_address(options.socket_path);
    if (::bind(listener.value, reinterpret_cast<const sockaddr*>(&address),
               sizeof(address)) != 0) {
        error = std::string("bind: ") + std::strerror(errno);
        return 1;
    }
    if (::listen(listener.value, std::max(16, options.parallel * 4)) != 0) {
        error = std::string("listen: ") + std::strerror(errno);
        return 1;
    }
    std::fprintf(stderr, "listening on %s\n", options.socket_path.c_str());

    while (true) {
        const int connection = ::accept(listener.value, nullptr, nullptr);
        if (connection < 0) {
            if (errno == EINTR) continue;
            error = std::string("accept: ") + std::strerror(errno);
            return 1;
        }
        std::thread(handle, connection, manager.get(), options.context).detach();
    }
}

int run_client(const ClientOptions& options, std::string& error) {
    std::vector<int> prompt;
    std::string word;
    while (std::cin >> word) {
        int token = -1;
        if (!integer(word, token) || token < 0) {
            error = "stdin contains an invalid token ID";
            return 1;
        }
        prompt.push_back(token);
    }
    if (prompt.empty()) {
        error = "stdin contains no token IDs";
        return 1;
    }

    Fd socket(unix_socket(options.socket_path, error));
    if (socket.value < 0) return 1;
    const sockaddr_un address = unix_address(options.socket_path);
    if (::connect(socket.value, reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) != 0) {
        error = std::string("connect: ") + std::strerror(errno);
        return 1;
    }

    std::ostringstream request;
    request << "POST /infer\n"
            << "max-tokens: " << options.max_tokens << "\n\n";
    for (size_t index = 0; index < prompt.size(); ++index) {
        if (index) request << ' ';
        request << prompt[index];
    }
    request << "\n\n";
    if (!send_all(socket.value, request.str())) {
        error = "cannot send request";
        return 1;
    }

    Reader reader(socket.value);
    std::string line;
    int status = 0;
    if (!reader.line(line)) {
        error = "incomplete response";
        return 1;
    }
    std::istringstream first(line);
    if (!(first >> status) || status < 100 || status > 599) {
        error = "invalid response status";
        return 1;
    }
    std::string message;
    while (reader.line(line) && !line.empty()) {
        if (line.compare(0, 9, "message: ") == 0) message = line.substr(9);
    }
    if (!line.empty()) {
        error = "incomplete response parameters";
        return 1;
    }
    if (status != 200) {
        error = message.empty() ? "server error" : message;
        return 1;
    }
    while (reader.line(line)) {
        if (line.empty()) return 0;
        std::cout << line << '\n';
        std::cout.flush();
    }
    error = "incomplete response";
    return 1;
}

bool option_value(int& index, int argc, char** argv,
                  std::string& value, std::string& error) {
    if (++index >= argc) {
        error = "missing option value";
        return false;
    }
    value = argv[index];
    return true;
}

bool server_options(int argc, char** argv, ServerOptions& options,
                    std::string& error) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--mock") {
            options.mock = true;
            continue;
        }
        if (argument != "-l" && argument != "-m" &&
            argument != "--parallel" && argument != "--context") {
            error = "unknown option: " + argument;
            return false;
        }
        std::string value;
        if (!option_value(index, argc, argv, value, error)) return false;
        if (argument == "-l") {
            options.socket_path = value;
        } else if (argument == "-m") {
            options.model_path = value;
        } else if (argument == "--parallel") {
            if (!integer(value, options.parallel) || options.parallel <= 0) {
                error = "invalid --parallel";
                return false;
            }
        } else if (argument == "--context") {
            if (!integer(value, options.context) || options.context <= 0) {
                error = "invalid --context";
                return false;
            }
        }
    }
    if (options.socket_path.empty()) {
        error = "-l is required";
        return false;
    }
    if (options.model_path.empty()) {
        error = "-m is required";
        return false;
    }
    return true;
}

bool client_options(int argc, char** argv, ClientOptions& options,
                    std::string& error) {
    if (argc < 2) {
        error = "socket path is required";
        return false;
    }
    options.socket_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument != "-n") {
            error = "unknown option: " + argument;
            return false;
        }
        std::string value;
        if (!option_value(index, argc, argv, value, error)) return false;
        if (!integer(value, options.max_tokens) || options.max_tokens <= 0) {
            error = "invalid -n";
            return false;
        }
    }
    return true;
}

void usage(const char* program) {
    std::fprintf(stderr,
        "server: %s -l SOCKET -m MODEL [--parallel N] [--context N] [--mock]\n"
        "client: %s SOCKET [-n N]\n",
        program, program);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    std::string error;
    int result = 1;
    if (std::string(argv[1]) == "-l") {
        ServerOptions options;
        if (server_options(argc, argv, options, error)) {
            result = run_server(options, error);
        }
    } else {
        ClientOptions options;
        if (client_options(argc, argv, options, error)) {
            result = run_client(options, error);
        }
    }
    if (result != 0) std::fprintf(stderr, "qwen35: %s\n", error.c_str());
    return result;
}
