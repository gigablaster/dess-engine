struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

struct Material {
    float emissive_power;
    float alpha_cut;
};

[[vk::binding(1, 0)]] SamplerState base_sampler;
[[vk::binding(0, 1)]] Texture2D<float4> base;
[[vk::binding(1, 1)]] Texture2D<float4> normals;
[[vk::binding(2, 1)]] Texture2D<float4> metallic_roughness;
[[vk::binding(3, 1)]] Texture2D<float4> occlusion;
[[vk::binding(4, 1)]] Texture2D<float4> emissive;
[[vk::binding(5, 1)]] ConstantBuffer<Material> material;

float4 main(VsOut psin) : SV_TARGET {
    float4 color = base.Sample(base_sampler, psin.uv);

    return color;
}