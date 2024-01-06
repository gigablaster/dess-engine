#include "include/bdrf.hlsl"

struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float2 uv1: TEXCOORD1;
    [[vk::location(2)]] float2 uv2: TEXCOORD2;
    [[vk::location(3)]] float3x3 tangent_basis: TBASIS;
};

struct Material {
    float emissive_power;
    float alpha_cut;
};

[[vk::binding(0, 0)]] ConstantBuffer<PassData> pass_data;
[[vk::binding(1, 0)]] ConstantBuffer<LightData> lights;
[[vk::binding(32, 0)]] SamplerState base_sampler;
[[vk::binding(0, 1)]] Texture2D<float4> base;
[[vk::binding(1, 1)]] Texture2D<float4> normals;
[[vk::binding(2, 1)]] Texture2D<float4> metallic_roughness;
[[vk::binding(3, 1)]] Texture2D<float4> occlusion;
[[vk::binding(4, 1)]] Texture2D<float4> emissive;
[[vk::binding(5, 1)]] ConstantBuffer<Material> material;

float3 unpack_normal(float4 tx) {
    float2 normal_xy = tx.xy * 2.0 - 1.0;
    float normal_z = sqrt(saturate(1.0 - dot(normal_xy, normal_xy)));
    return float3(normal_xy, normal_z);
}


float4 main(VsOut psin) : SV_TARGET {
    float3 albedo = base.Sample(base_sampler, psin.uv1).rgb;
    float3 mr = metallic_roughness.Sample(base_sampler, psin.uv1).rgb;
    float ao = occlusion.Sample(base_sampler, psin.uv1).r;
    float metallic = mr.b;
    float roughness = mr.g;
    float3 normal = unpack_normal(normals.Sample(base_sampler, psin.uv1));

    roughness = max(roughness, 0.0001);

    float3 N = normalize(mul(psin.tangent_basis, normal));
    float3 V = normalize(pass_data.eye_position - psin.pos);
    
    float3 f0 = float3(0.04, 0.04, 0.04);
    f0 = lerp(f0, albedo, metallic);

    float3 Lo = float3(0, 0, 0);
    for (int i = 0; i < 3; i++) {
        Lo += bdrf(N, V, albedo, f0, metallic, roughness, ao, lights.dir[i].direction, lights.dir[i].color);
    }
    float3 diffuse_ambient = ambient_light(N, lights.ambient);
    float3 specular_ambient = ambient_light(-reflect(V, N), lights.ambient);
    float3 ambient = lerp(diffuse_ambient * albedo * ao, lerp(specular_ambient, diffuse_ambient, roughness * roughness) * f0, metallic);
    float3 color = ambient + Lo;

    return float4(color, 1.0);
}