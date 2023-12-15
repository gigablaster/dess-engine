struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

[[vk::binding(1, 0)]] SamplerState base_sampler;
[[vk::binding(0, 1)]] Texture2D<float4> base;

float4 main(VsOut psin) : SV_TARGET {
    float4 color = base.Sample(base_sampler, psin.uv);

    return color;
}