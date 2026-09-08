// GPU fixtures for the native affine4/8 kernels and recurrent state transition.
#include "arch/mlx/engine.cpp"
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
    using namespace q3x_backend;
    // Run fatal paths in separate processes; no real model weights are needed.
    if(argc==2) {
        char error[1024];
        if(!q3x_internal::log_configure(Q3X_LOG_ERROR,nullptr,1024*1024,1,
                                       error,sizeof(error))) return 98;
        State state(32);
        if(std::strcmp(argv[1],"--fail-forward")==0) {
            Model model;  // Invalid scalar weights force an MLX argument exception.
            const int token=0;
            state_forward(&model,&state,&token,1,true);
        } else if(std::strcmp(argv[1],"--fail-restore")==0) {
            state_checkpoint_restore(&state);  // No saved checkpoint.
        }
        return 99;  // Neither fatal path may return to its caller.
    }
    mx::set_default_device(mx::Device::gpu);
    mx::metal::set_metallib_path(Q3X_MLX_METALLIB);
    for(int bits:{4,8}) {
        std::vector<float> data(3*8*128);
        for(size_t i=0;i<data.size();++i)data[i]=float(int(i%97)-48)/32;
        auto w=array(data.data(),{3,8,128});
        auto packed=mx::quantize(mx::astype(w,mx::bfloat16),64,bits);
        auto dense=mx::dequantize(packed[0],packed[1],packed[2],64,bits);
        auto x=mx::ones({2,1,128},mx::bfloat16);
        auto ids=array({2,0},mx::uint32);
        auto got=mx::gather_qmm(x,packed[0],packed[1],packed[2],{},ids,true,64,bits);
        auto expected=mx::matmul(x,mx::swapaxes(mx::take(dense,ids,0),-1,-2));
        auto e=mx::max(mx::abs(mx::astype(got,mx::float32)-mx::astype(expected,mx::float32)));
        mx::eval(e);
        if(e.item<float>()>0.25f)throw std::runtime_error("indexed affine qmm fixture");
        std::cout<<"affine bits="<<bits<<" max_error="<<e.item<float>()<<'\n';
    }
    const int t=5;
    auto q=mx::full({1,t,C.KH,C.KD},1.0f/128,mx::bfloat16);
    auto k=mx::full({1,t,C.KH,C.KD},1.0f/16,mx::bfloat16);
    auto v=mx::full({1,t,C.VH,C.VD},0.5f,mx::bfloat16);
    auto g=mx::full({1,t,C.VH},0.9375f,mx::float32);
    auto beta=mx::full({1,t,C.VH},0.5f,mx::bfloat16);
    auto state=mx::zeros({1,C.VH,C.VD,C.KD},mx::float32);
    auto kernel=mx::fast::metal_kernel("q3x_delta_fixture",{"q","k","v","g","beta","state_in","T"},
                                      {"y","state_out"},delta_source);
    auto result=kernel({q,k,v,g,beta,state,array(t)},{{1,t,C.VH,C.VD},state.shape()},
                       {mx::bfloat16,mx::float32},{32,C.VD,C.VH},{32,4,1},
                       {{"InT",mx::bfloat16},{"Dk",C.KD},{"Dv",C.VD},{"Hk",C.KH},{"Hv",C.VH}},
                       {},false,{});
    auto out=mx::astype(result[0],mx::float32);mx::eval(out,result[1]);
    float cell=0;
    for(int i=0;i<t;++i) {
        cell*=0.9375f;
        float d=(0.5f-cell*C.KD/16)*0.5f;
        cell+=d/16;
        if(std::abs(out.data<float>()[i*C.VH*C.VD]-cell)>0.001f)
            throw std::runtime_error("delta output fixture");
    }
    if(std::abs(result[1].data<float>()[0]-cell)>1e-7)
        throw std::runtime_error("delta state fixture");
    char err[1024];
    if(model_create("/nonexistent/qwen3x-mlx-fixture",err,sizeof(err)) || !std::strstr(err,"cannot open"))
        throw std::runtime_error("load exception boundary fixture");
    std::cout<<"delta FP32 state and exception boundary passed\n";
}
