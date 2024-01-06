#include "include/bdrf.hlsl"

struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float3 world_position: TEXCOORD0;
};

[[vk::binding(0, 0)]] ConstantBuffer<PassData> pass_data;
[[vk::binding(1, 0)]] ConstantBuffer<LightData> lights;

float4 main(VsOut psin) : SV_TARGET {
    float3 V = normalize(pass_data.eye_position - psin.world_position);
    float3 color = ambient_light(-V, lights.ambient);

    return float4(color, 1.0);
}