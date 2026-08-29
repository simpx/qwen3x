# Third-party code

`nlohmann/json.hpp` is the single-header distribution of
[`nlohmann/json`](https://github.com/nlohmann/json), pinned to v3.12.0.
It is used only by `parser.cpp` and is distributed under the accompanying
MIT license.

```text
json.hpp SHA-256:
aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
```

`httplib/httplib.h` is [`cpp-httplib`](https://github.com/yhirose/cpp-httplib)
v0.54.0 at commit `9185fdb6ae4f7aaa3a78a3ec7289c606c7c5d951`. It provides
the HTTP/1.1 server and is distributed under `httplib/LICENSE`.

`spdlog/` is [`spdlog`](https://github.com/gabime/spdlog) v1.17.0 at commit
`79524ddd08a4ec981b7fea76afd08ee05f83755d`. It is compiled into the
executable with its bundled fmt implementation and is distributed under
`spdlog/LICENSE`.
