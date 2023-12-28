struct PassData {
    float4x4 view;
    float4x4 projection;
};

struct ObjectData {
    float position_min_range;
    float position_max_range;
    float uv1_min_range;
    float uv1_max_range;
    float uv2_min_range;
    float uv2_max_range;
};

struct InstanceTransform {
    float4x4 model;
};

struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

struct VsIn {
    [[vk::location(0)]] float3 pos: POSITION;
    [[vk::location(1)]] float2 uv: TEXCOORD0;
    uint instance: SV_InstanceID;
};

[[vk::binding(0, 0)]] ConstantBuffer<PassData> pass_data;
[[vk::binding(0, 2)]] ConstantBuffer<ObjectData> object_data;
[[vk::binding(0, 3)]] StructuredBuffer<InstanceTransform> transforms;

VsOut main(VsIn vsin) {
    VsOut vsout;
    
    float4x4 mvp = transforms[vsin.instance].model * pass_data.view * pass_data.projection;
    vsout.position = mul(mvp, float4(vsin.pos, 0.0));
    vsout.uv = vsin.uv;

    return vsout;
}