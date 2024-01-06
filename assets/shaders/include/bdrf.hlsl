#include "common.hlsl"

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

float3 ambient_light(float3 dir, HemisphericalAmbient ambient) {
    float f = dot(dir, float3(0, 1, 0));
    float3 target = f < 0.0 ? ambient.bottom : ambient.top;
    return lerp(ambient.middle, target, abs(f));
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
