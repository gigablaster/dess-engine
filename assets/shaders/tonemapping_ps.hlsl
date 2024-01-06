struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

struct Params {
    float expouse;
};

[[vk::binding(0, 0)]] Texture2D<float4> hdr;
[[vk::binding(1, 0)]] ConstantBuffer<Params> params;
[[vk::binding(32, 0)]] SamplerState base_sampler;

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 reinhard_jodie(float3 v)
{
    float l = luminance(v);
    float3 tv = v / (1.0f + v);
    return lerp(v / (1.0f + l), tv, tv);
}

float4 main(VsOut psin) : SV_TARGET {
    float3 hdr_color = hdr.Sample(base_sampler, psin.uv).xyz;
    float3 expoused = 1.0 - exp(-hdr_color * params.expouse);
    float3 ldr_color = reinhard_jodie(expoused);
    float3 srgb = pow(ldr_color, 1/2.2);
    return float4(srgb, 1.0);
}