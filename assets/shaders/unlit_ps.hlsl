struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

[[vk::binding(1, 1)]] Texture2D main_texture;
[[vk::binding(32)]] SamplerState main_sampler;

float4 main(VsOut psin) : SV_TARGET {
    float4 color = main_texture.Sample(main_sampler, psin.uv);

    return color;
}