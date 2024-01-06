#include "include/common.hlsl"

struct VsIn {
    [[vk::location(0)]] float3 position: POSITION;
    uint id: SV_INSTANCEID;
};

struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float3 world_position: TEXCOORD0;
};

[[vk::binding(0, 0)]] ConstantBuffer<PassData> pass_data;
[[vk::binding(0, 3)]] StructuredBuffer<InstanceTransform> transforms;

VsOut main(VsIn vsin) {
    VsOut vsout;

    float4x4 model = transforms[vsin.id].model;
    float4x4 mvp = mul(pass_data.projection, mul(pass_data.view, model));
    float3 world_position = mul(model, float4(vsin.position, 1.0)).xyz;
    float4 pos = mul(mvp, float4(vsin.position, 1.0));

    vsout.world_position = world_position;
    vsout.position = pos;

    return vsout;
}