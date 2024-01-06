struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D<float4> hdr;
[[vk::binding(32, 0)]] SamplerState base_sampler;

float4 main(VsOut psin) : SV_TARGET {
    float3 color = hdr.Sample(base_sampler, psin.uv).xyz;
    return float4(color, 1.0);
}