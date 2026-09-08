// MLX implementation of the Qwen3.6 text forward. The C++17 runtime sees only
// internal.h; MLX types, GPU scheduling and exception handling live here.
#include "internal.h"
#include "model_config.h"
#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/compile.h"
#include "mlx/memory.h"
#include "mlx/backend/metal/metal.h"
#include "third_party/nlohmann/json.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;
using mx::array;
using Json = nlohmann::json;
namespace q3x_backend {
namespace {
constexpr auto C = q3x_model::QWEN36_35B_A3B;
constexpr float EPS = 1e-6f;
std::recursive_mutex gpu_mutex;
void load_error(const char* message, char* err, size_t errlen) noexcept {
    if(err && errlen) std::snprintf(err, errlen, "MLX: %s", message);
    LOG_ERROR("MLX model load failed: %s", message);
}
[[noreturn]] void fatal(const char* operation, const char* message) noexcept {
    LOG_ERROR("MLX %s failed: %s", operation, message);
    std::abort();
}
array part(const array& a, int axis, int start, int stop) {
    mx::Shape lo(a.ndim(), 0), hi = a.shape();
    if (axis < 0) axis += a.ndim();
    lo[axis] = start; hi[axis] = stop;
    return mx::slice(a, lo, hi);
}
array silu(const array& x) { return x * mx::sigmoid(x); }
auto conv_silu = mx::compile([](const std::vector<array>& a) {
    return std::vector<array>{silu(a[0])};
}, true);
// Match mlx-lm's compiled activation precision, including intermediate rounding.
auto swiglu = mx::compile([](const std::vector<array>& a) {
    return std::vector<array>{silu(a[0]) * a[1]};
}, true);
auto gated_norm = mx::compile([](const std::vector<array>& a) {
    return std::vector<array>{mx::astype(silu(mx::astype(a[1], mx::float32)) *
                                       mx::astype(a[0], mx::float32), a[0].dtype())};
}, true);
auto decay = mx::compile([](const std::vector<array>& a) {
    auto t = a[1] + a[2];
    auto softplus = mx::logaddexp(t, array(0.0f,t.dtype()));
    return std::vector<array>{mx::exp(-mx::exp(mx::astype(a[0], mx::float32)) * softplus)};
}, true);

// Scalar-gate GatedDeltaNet recurrence, adapted from mlx-lm 0.31.3
// models/gated_delta.py (Apple Inc., MIT; see arch/mlx/NOTICE).
const char* delta_source = R"metal(
    auto hv = thread_position_in_grid.z;
    auto hk = hv / (Hv / Hk);
    auto lane = thread_position_in_threadgroup.x;
    auto dv = thread_position_in_grid.y;
    constexpr int npt = Dk / 32;
    float state[npt];
    for (int i = 0; i < npt; ++i)
        state[i] = state_in[(hv * Dv + dv) * Dk + lane * npt + i];
    for (int t = 0; t < T; ++t) {
        float mem = 0;
        for (int i = 0; i < npt; ++i) {
            int d = lane * npt + i;
            state[i] *= g[t * Hv + hv];
            mem += state[i] * k[(t * Hk + hk) * Dk + d];
        }
        mem = simd_sum(mem);
        float delta = (v[(t * Hv + hv) * Dv + dv] - mem) * beta[t * Hv + hv];
        float out = 0;
        for (int i = 0; i < npt; ++i) {
            int d = lane * npt + i;
            state[i] += k[(t * Hk + hk) * Dk + d] * delta;
            out += state[i] * q[(t * Hk + hk) * Dk + d];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) y[(t * Hv + hv) * Dv + dv] = InT(out);
    }
    for (int i = 0; i < npt; ++i)
        state_out[(hv * Dv + dv) * Dk + lane * npt + i] = state[i];
)metal";

using Weights = std::unordered_map<std::string, array>;
array take_weight(Weights& weights, const std::string& name, const mx::Shape& shape) {
    auto it = weights.find(name);
    if (it == weights.end()) throw std::runtime_error("missing tensor " + name);
    if (it->second.shape() != shape) throw std::runtime_error("wrong shape " + name);
    auto out = it->second;
    weights.erase(it);
    return out;
}
struct Linear {
    array w{0}, scales{0}, biases{0};
    int bits = 4;
    void load(Weights& all, const Json& quant, const std::string& name,
              int rows, int cols, int experts = 0) {
        bits = quant.contains(name) ? quant.at(name).at("bits").get<int>() : 4;
        int group = quant.contains(name) ? quant.at(name).at("group_size").get<int>() : 64;
        if (quant.contains(name) && quant.at(name).value("mode",std::string("affine"))!="affine")
            throw std::runtime_error("unsupported quantization mode " + name);
        if (group != 64 || (bits != 4 && bits != 8))
            throw std::runtime_error("unsupported quantization " + name);
        mx::Shape ws{rows, cols * bits / 32}, ss{rows, cols / 64};
        if (experts) { ws.insert(ws.begin(), experts); ss.insert(ss.begin(), experts); }
        w = take_weight(all, name + ".weight", ws);
        scales = take_weight(all, name + ".scales", ss);
        biases = take_weight(all, name + ".biases", ss);
        if (w.dtype() != mx::uint32 || scales.dtype() != mx::bfloat16 ||
            biases.dtype() != mx::bfloat16)
            throw std::runtime_error("unexpected quantized dtype " + name);
    }
    array operator()(const array& x) const {
        return mx::quantized_matmul(x, w, scales, biases, true, 64, bits);
    }
    array gather(const array& x, const array& ids, bool sorted) const {
        return mx::gather_qmm(x, w, scales, biases, {}, ids, true, 64, bits, "affine", sorted);
    }
    array embed(const array& tokens) const {
        return mx::dequantize(mx::take(w, tokens, 0), mx::take(scales, tokens, 0),
                              mx::take(biases, tokens, 0), 64, bits);
    }
};
struct Layer {
    array norm{0}, post_norm{0}, qnorm{0}, knorm{0}, conv{0}, a_log{0}, dt{0}, dnorm{0};
    Linear q, k, v, out, z, a, b;
    Linear router, gate, up, down, shared_gate, shared_up, shared_down, shared_route;
};
struct Cache {
    array first{0}, second{0};  // conv/recurrent or K/V; arrays are immutable snapshots
    bool initialized = false;
};
Json read_json(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    return Json::parse(file);
}
Json packed_config(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    uint64_t length=0;
    if(!file.read(reinterpret_cast<char*>(&length),8) || length==0 || length>1024*1024)
        throw std::runtime_error("invalid MLX model header");
    std::string text(length,'\0');
    if(!file.read(text.data(),length))throw std::runtime_error("truncated MLX header");
    auto header=Json::parse(text);
    const auto& meta=header.at("__metadata__");
    if(meta.at("qwen3x_format")!="mlx-affine-v1" || meta.at("model_id")!="36035")
        throw std::runtime_error("unsupported MLX model format");
    std::vector<std::pair<uint64_t,uint64_t>> ranges;
    for(auto& item:header.items()) {
        if(item.key()=="__metadata__")continue;
        if(item.key().rfind("language_model.",0)!=0)throw std::runtime_error("non-text packed tensor");
        auto off=item.value().at("data_offsets").get<std::vector<uint64_t>>();
        if(off.size()!=2 || off[1]<off[0])throw std::runtime_error("invalid packed tensor offsets");
        ranges.emplace_back(off[0],off[1]);
    }
    std::sort(ranges.begin(),ranges.end());
    uint64_t end=0;
    for(auto [lo,hi]:ranges) {
        if(lo!=end)throw std::runtime_error("overlapping or missing packed tensor data");
        end=hi;
    }
    auto bytes=std::filesystem::file_size(path);
    if(bytes<length+8 || end!=bytes-length-8)throw std::runtime_error("MLX model EOF mismatch");
    // MLX's safetensors reader also verifies dtype, shape product and each range.
    return Json::parse(meta.at("config").get<std::string>());
}
} // namespace

struct Model {
    Linear embedding, head;
    array norm{0};
    std::vector<Layer> layers;
    int chunk = 2048;
};
struct State {
    std::vector<Cache> cache, saved;
    array logits{0}, saved_logits{0};
    int position = 0, saved_position = -1, capacity;
    explicit State(int context) : cache(C.N), capacity(context) {}
};

static array attention(const Layer& l, Cache& cache, const array& x, int offset) {
    int t = x.shape(1);
    auto qg = mx::reshape(l.q(x), {1,t,C.AH,2*C.AD});
    auto q = mx::transpose(mx::fast::rms_norm(part(qg,-1,0,C.AD),l.qnorm,EPS), {0,2,1,3});
    auto gate = mx::reshape(part(qg,-1,C.AD,2*C.AD), {1,t,C.AH*C.AD});
    auto k = mx::transpose(mx::fast::rms_norm(mx::reshape(l.k(x),{1,t,C.KVH,C.AD}),l.knorm,EPS),{0,2,1,3});
    auto v = mx::transpose(mx::reshape(l.v(x),{1,t,C.KVH,C.AD}),{0,2,1,3});
    q = mx::fast::rope(q,C.RD,false,10000000.0f,1.0f,offset);
    k = mx::fast::rope(k,C.RD,false,10000000.0f,1.0f,offset);
    // Match mlx-lm's 256-token capacity layout. MLX slice_update is functional:
    // a live checkpoint keeps its old storage; unshared buffers can be donated.
    if (!cache.initialized || offset+t>cache.first.shape(2)) {
        int extra=((t+255)/256)*256;
        auto nk=mx::zeros({1,C.KVH,extra,C.AD},k.dtype());
        auto nv=mx::zeros({1,C.KVH,extra,C.AD},v.dtype());
        if(cache.initialized) {
            nk=mx::concatenate({part(cache.first,2,0,offset),nk},2);
            nv=mx::concatenate({part(cache.second,2,0,offset),nv},2);
        }
        cache.first=nk;cache.second=nv;
    }
    cache.first=mx::slice_update(cache.first,k,{0,0,offset,0},{1,C.KVH,offset+t,C.AD});
    cache.second=mx::slice_update(cache.second,v,{0,0,offset,0},{1,C.KVH,offset+t,C.AD});
    cache.initialized=true;
    k=part(cache.first,2,0,offset+t);v=part(cache.second,2,0,offset+t);
    auto y = mx::fast::scaled_dot_product_attention(q,k,v,1.0f/16.0f,t>1?"causal":"");
    y = mx::reshape(mx::transpose(y,{0,2,1,3}),{1,t,C.AH*C.AD});
    return l.out(y * mx::sigmoid(gate));
}

static array delta(const Layer& l, Cache& cache, const array& x) {
    int t=x.shape(1), kd=C.KH*C.KD, vd=C.VH*C.VD, channels=2*kd+vd;
    auto qkv=l.q(x), z=mx::reshape(l.z(x),{1,t,C.VH,C.VD});
    auto beta=mx::sigmoid(l.b(x));
    auto g=decay({l.a_log,l.a(x),l.dt})[0];
    if (!cache.initialized) {
        cache.first=mx::zeros({1,C.CK-1,channels},x.dtype());
        cache.second=mx::zeros({1,C.VH,C.VD,C.KD},mx::float32);
    }
    auto input=mx::concatenate({cache.first,qkv},1);
    cache.first=mx::contiguous(part(input,1,t,t+C.CK-1));
    auto conv=conv_silu({mx::conv1d(input,l.conv,1,0,1,channels)})[0];
    auto q=mx::reshape(part(conv,-1,0,kd),{1,t,C.KH,C.KD});
    auto k=mx::reshape(part(conv,-1,kd,2*kd),{1,t,C.KH,C.KD});
    auto v=mx::reshape(part(conv,-1,2*kd,channels),{1,t,C.VH,C.VD});
    q=mx::fast::rms_norm(q,{},EPS)*array(1.0f/C.KD,q.dtype());
    k=mx::fast::rms_norm(k,{},EPS)*array(1.0f/std::sqrt(float(C.KD)),k.dtype());
    static auto kernel=mx::fast::metal_kernel("q3x_gated_delta",{"q","k","v","g","beta","state_in","T"},
                                             {"y","state_out"},delta_source);
    auto result=kernel({q,k,v,g,beta,cache.second,array(t)},
                       {{1,t,C.VH,C.VD},{1,C.VH,C.VD,C.KD}}, {q.dtype(),mx::float32},
                       {32,C.VD,C.VH},{32,4,1},
                       {{"InT",q.dtype()},{"Dk",C.KD},{"Dv",C.VD},{"Hk",C.KH},{"Hv",C.VH}},
                       {},false,{});
    cache.second=result[1]; cache.initialized=true;
    auto y=mx::fast::rms_norm(result[0],l.dnorm,EPS);
    return l.out(mx::reshape(gated_norm({y,z})[0],{1,t,vd}));
}

static array moe(const Layer& l, const array& x) {
    int t=x.shape(1);
    auto probs=mx::softmax(l.router(x),-1,true);
    auto ids=part(mx::argpartition(probs,-C.top_k,-1),-1,C.experts-C.top_k,C.experts);
    auto scores=mx::take_along_axis(probs,ids,-1);
    scores=scores/mx::sum(scores,-1,true);
    bool sorted=t*C.top_k>=64;
    array input=mx::reshape(x,{1,t,1,1,C.H}), indices=ids, inverse{0};
    if(sorted) {
        auto flat=mx::reshape(ids,{-1});
        auto order=mx::argsort(flat);
        inverse=mx::argsort(order);
        indices=mx::take(flat,order,0);
        input=mx::take(mx::reshape(x,{t,1,C.H}),mx::floor_divide(order,array(C.top_k)),0);
    }
    auto y=l.down.gather(swiglu({l.gate.gather(input,indices,sorted),l.up.gather(input,indices,sorted)})[0],indices,sorted);
    if(sorted) y=mx::take(y,inverse,0);
    y=mx::reshape(y,{1,t,C.top_k,C.H});
    y=mx::sum(y*mx::expand_dims(scores,-1),-2);
    auto shared=l.shared_down(swiglu({l.shared_gate(x),l.shared_up(x)})[0]);
    return y + mx::sigmoid(l.shared_route(x))*shared;
}

static void forward(Model& m, State& s, const int* tokens, int count, bool logits) {
    for(int begin=0;begin<count;begin+=m.chunk) {
        int n=std::min(m.chunk,count-begin);
        auto x=m.embedding.embed(array(tokens+begin,{1,n},mx::int32));
        for(int i=0;i<C.N;++i) {
            auto& l=m.layers[i];
            auto h=mx::fast::rms_norm(x,l.norm,EPS);
            x=x+((i+1)%C.AI ? delta(l,s.cache[i],h) : attention(l,s.cache[i],h,s.position));
            x=x+moe(l,mx::fast::rms_norm(x,l.post_norm,EPS));
        }
        std::vector<array> ready;
        if(logits && begin+n==count) {
            s.logits=mx::astype(m.head(mx::fast::rms_norm(part(x,1,n-1,n),m.norm,EPS)),mx::float32);
            ready.push_back(s.logits);
        }
        for(auto& c:s.cache) {ready.push_back(c.first);ready.push_back(c.second);}
        mx::eval(ready);
        s.position+=n;
    }
}

Model* model_create(const char* path,char* err,size_t errlen) {
    std::lock_guard<std::recursive_mutex> lock(gpu_mutex);
    try {
        mx::set_default_device(mx::Device::gpu);
        const char* resource=std::getenv("Q3X_MLX_METALLIB");
        mx::metal::set_metallib_path(resource?resource:Q3X_MLX_METALLIB);
        mx::set_cache_limit(size_t(1)<<30);
        auto directory=std::filesystem::path(path);
        bool packed=std::filesystem::is_regular_file(directory) && directory.filename()!="config.json";
        std::string packed_name=directory.filename().string();
        Json config=packed?packed_config(directory):Json();
        if(packed)directory=directory.parent_path();
        if(directory.filename()=="config.json") directory=directory.parent_path();
        if(!packed)config=read_json(directory/"config.json");
        const auto& tc=config.at("text_config");
        for(auto [key,value]:std::vector<std::pair<std::string,int>>{
            {"hidden_size",C.H},{"num_hidden_layers",C.N},{"num_attention_heads",C.AH},
            {"num_key_value_heads",C.KVH},{"head_dim",C.AD},{"vocab_size",C.V},
            {"num_experts",C.experts},{"num_experts_per_tok",C.top_k},{"moe_intermediate_size",C.I},
            {"shared_expert_intermediate_size",C.shared_I},
            {"full_attention_interval",C.AI},{"linear_num_key_heads",C.KH},{"linear_num_value_heads",C.VH},
            {"linear_key_head_dim",C.KD},{"linear_value_head_dim",C.VD},{"linear_conv_kernel_dim",C.CK}})
            if(tc.at(key)!=value) throw std::runtime_error("unsupported model shape: "+key);
        const auto& rope=tc.at("rope_parameters");
        if(config.at("model_type")!="qwen3_5_moe" || tc.at("rms_norm_eps").get<float>()!=EPS ||
           tc.at("tie_word_embeddings").get<bool>() || tc.at("attention_bias").get<bool>() ||
           !tc.at("attn_output_gate").get<bool>() || rope.at("rope_theta").get<float>()!=10000000.0f ||
           rope.at("partial_rotary_factor").get<float>()!=0.25f ||
           rope.value("rope_type",std::string("default"))!="default")
            throw std::runtime_error("unsupported Qwen3.6 text configuration");
        auto quant=config.at("quantization");
        if(quant.at("bits")!=4 || quant.at("group_size")!=64 || quant.at("mode")!="affine")
            throw std::runtime_error("expected MLX affine4/group64 checkpoint");
        std::set<std::string> shards;
        if(packed)shards.insert(packed_name);
        else {
            auto index=read_json(directory/"model.safetensors.index.json");
            for(auto& item:index.at("weight_map").items()) shards.insert(item.value().get<std::string>());
        }
        Weights all;
        for(const auto& shard:shards) {
            if(std::filesystem::path(shard).filename()!=shard) throw std::runtime_error("invalid shard path");
            auto loaded=mx::load_safetensors((directory/shard).string());
            std::vector<array> ready;
            for(auto& [name,w]:loaded.first) {
                if(name.rfind("language_model.",0)!=0 || name.find(".mtp.")!=std::string::npos) continue;
                if(!all.emplace(name,w).second) throw std::runtime_error("duplicate tensor "+name);
                ready.push_back(w);
            }
            mx::eval(ready);
            LOG_INFO("MLX loaded shard=%s active_mib=%.1f",shard.c_str(),mx::get_active_memory()/1048576.0);
        }
        auto m=std::make_unique<Model>();
        if(const char* c=std::getenv("Q3X_MLX_CHUNK")) {
            m->chunk=std::atoi(c);
            if(m->chunk<1 || m->chunk>4096) throw std::runtime_error("Q3X_MLX_CHUNK outside 1..4096");
        }
        std::string root="language_model.model.";
        m->embedding.load(all,quant,root+"embed_tokens",C.V,C.H);
        m->head.load(all,quant,"language_model.lm_head",C.V,C.H);
        m->norm=take_weight(all,root+"norm.weight",{C.H});
        m->layers.resize(C.N);
        for(int i=0;i<C.N;++i) {
            auto& l=m->layers[i]; std::string p=root+"layers."+std::to_string(i)+".";
            l.norm=take_weight(all,p+"input_layernorm.weight",{C.H});
            l.post_norm=take_weight(all,p+"post_attention_layernorm.weight",{C.H});
            if((i+1)%C.AI) {
                auto d=p+"linear_attn.";
                l.q.load(all,quant,d+"in_proj_qkv",2*C.KH*C.KD+C.VH*C.VD,C.H);
                l.z.load(all,quant,d+"in_proj_z",C.VH*C.VD,C.H);
                l.a.load(all,quant,d+"in_proj_a",C.VH,C.H); l.b.load(all,quant,d+"in_proj_b",C.VH,C.H);
                l.out.load(all,quant,d+"out_proj",C.H,C.VH*C.VD);
                l.conv=take_weight(all,d+"conv1d.weight",{2*C.KH*C.KD+C.VH*C.VD,C.CK,1});
                l.a_log=take_weight(all,d+"A_log",{C.VH}); l.dt=take_weight(all,d+"dt_bias",{C.VH});
                l.dnorm=take_weight(all,d+"norm.weight",{C.VD});
            } else {
                auto a=p+"self_attn.";
                l.q.load(all,quant,a+"q_proj",2*C.AH*C.AD,C.H);
                l.k.load(all,quant,a+"k_proj",C.KVH*C.AD,C.H); l.v.load(all,quant,a+"v_proj",C.KVH*C.AD,C.H);
                l.out.load(all,quant,a+"o_proj",C.H,C.AH*C.AD);
                l.qnorm=take_weight(all,a+"q_norm.weight",{C.AD}); l.knorm=take_weight(all,a+"k_norm.weight",{C.AD});
            }
            auto f=p+"mlp.";
            l.router.load(all,quant,f+"gate",C.experts,C.H);
            l.gate.load(all,quant,f+"switch_mlp.gate_proj",C.I,C.H,C.experts);
            l.up.load(all,quant,f+"switch_mlp.up_proj",C.I,C.H,C.experts);
            l.down.load(all,quant,f+"switch_mlp.down_proj",C.H,C.I,C.experts);
            l.shared_gate.load(all,quant,f+"shared_expert.gate_proj",C.shared_I,C.H);
            l.shared_up.load(all,quant,f+"shared_expert.up_proj",C.shared_I,C.H);
            l.shared_down.load(all,quant,f+"shared_expert.down_proj",C.H,C.shared_I);
            l.shared_route.load(all,quant,f+"shared_expert_gate",1,C.H);
        }
        if(!all.empty()) throw std::runtime_error("unexpected text tensor "+all.begin()->first);
        LOG_INFO("MLX model ready chunk=%d active_mib=%.1f",m->chunk,mx::get_active_memory()/1048576.0);
        return m.release();
    } catch(const std::exception& e) {load_error(e.what(),err,errlen);}
      catch(...) {load_error("unknown model load exception",err,errlen);}
    return nullptr;
}
void model_destroy(Model* m) {std::lock_guard<std::recursive_mutex> lock(gpu_mutex);delete m;}
State* state_create(Model*,int context) {
    std::lock_guard<std::recursive_mutex> lock(gpu_mutex);
    return new State(context);
}
void state_destroy(State* s) {std::lock_guard<std::recursive_mutex> lock(gpu_mutex);delete s;}
void state_reset(State* s) {
    std::lock_guard<std::recursive_mutex> lock(gpu_mutex);
    *s=State(s->capacity);
}
void state_forward(Model* m,State* s,const int* tokens,int count,bool logits) {
    std::lock_guard<std::recursive_mutex> lock(gpu_mutex);
    try {
        forward(*m,*s,tokens,count,logits);
        LOG_DEBUG("MLX forward position=%d count=%d active_bytes=%zu peak_bytes=%zu cache_bytes=%zu",
                  s->position,count,mx::get_active_memory(),mx::get_peak_memory(),mx::get_cache_memory());
    }
    catch(const std::exception& e){fatal("forward",e.what());}
    catch(...){fatal("forward","unknown exception");}
}
void state_checkpoint_save(State* s) {
    std::lock_guard<std::recursive_mutex> lock(gpu_mutex);
    s->saved=s->cache;s->saved_logits=s->logits;s->saved_position=s->position;
}
void state_checkpoint_restore(State* s) {
    std::lock_guard<std::recursive_mutex> lock(gpu_mutex);
    Q3X_ASSERT(s->saved_position>=0,"MLX checkpoint restore: no checkpoint");
    s->cache=s->saved;s->logits=s->saved_logits;s->position=s->saved_position;
}
int state_argmax(const State* s) {
    if(!s)return -1;
    // Forward materializes float32 logits; no GPU work can fail on this read.
    const auto* p=s->logits.data<float>();
    return int(std::max_element(p,p+C.V)-p);
}
void state_copy_logits(const State* s,float* out) {if(s)std::memcpy(out,s->logits.data<float>(),C.V*sizeof(float));}
uint32_t model_id(const Model*) {return C.id;}
int vocab_size(){return C.V;}
int max_context(){return q3x_model::MAX_CONTEXT;}
bool token_is_stop(int token){return token==248044 || token==248046;}
} // namespace q3x_backend

// Diagnostic adapter for offline benchmarks, not an HTTP endpoint. Darwin RSS
// alone does not account for all GPU allocations in unified memory.
extern "C" void q3x_mlx_memory_stats(uint64_t* out, bool reset_peak) {
    std::lock_guard<std::recursive_mutex> lock(q3x_backend::gpu_mutex);
    out[0]=mx::get_active_memory();out[1]=mx::get_peak_memory();out[2]=mx::get_cache_memory();
    if(reset_peak)mx::reset_peak_memory();
}
