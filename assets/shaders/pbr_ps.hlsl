struct PassData {
    float4x4 view;
    float4x4 projection;
    float3 eye_position;
};

struct DirectionalLight {
    float3 direction;
    float3 color;
};

struct HemisphericalAmbient {
    float3 top;
    float3 middle;
    float3 bottom;
};

struct LightData {
    DirectionalLight dir[3];
    HemisphericalAmbient ambient;
};

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

static const float PI = 3.1415926;

float distribution_ggx(float NdotH, float roughness) {
    float a = roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float geometry_shlick_ggx(float cos_theta, float k) {
    float num = cos_theta;
    float denom = cos_theta * (1.0 - k) + k;

    return num / denom;
}

float3 frensel_shlick(float cos_theta, float3 f0) {
    return f0 + (1.0 - f0) * pow(2, (-5.55473 * cos_theta - 6.98316 * cos_theta));
}

float geometry_smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggx1 = geometry_shlick_ggx(NdotV, k);
    float ggx2 = geometry_shlick_ggx(NdotL, k);

    return ggx1 * ggx2;
}

float3 ambient_light(float3 dir) {
    float f = dot(dir, float3(0, 1, 0));
    float3 target = f < 0.0 ? lights.ambient.bottom : lights.ambient.top;
    return lerp(lights.ambient.middle, target, abs(f));
}

float3 bdrf(float3 N, float3 V, float3 albedo, float3 f0, float metallic, float roughness, float ao, float3 light_dir, float3 light_color) {
    float3 L = normalize(light_dir);
    float3 H = normalize(L + V);

    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));
    float NdotV = saturate(dot(N, V));
    float HdotV = saturate(dot(H, V));

    float3 F = frensel_shlick(HdotV, f0);
    float NDF = distribution_ggx(NdotH, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);

    float3 num = NDF * G * F;
    float denom = 4.0 * NdotV * NdotL + 0.0001;
    float3 specular = num / denom;

    float3 kS = f0;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;

    kD *= 1.0  - metallic;

    return (kD * albedo * NdotL * ao + specular) * light_color;
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
    float3 diffuse_ambient = ambient_light(N);
    float3 specular_ambient = ambient_light(reflect(-V, N));
    float3 ambient = lerp(diffuse_ambient * albedo * ao, lerp(specular_ambient, diffuse_ambient, roughness * roughness) * f0, metallic);
    float3 color = ambient + Lo;

    return float4(pow(color, 1/2.2), 1.0);
}