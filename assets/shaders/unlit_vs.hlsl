struct PassData {
    float4x4 view;
    float4x4 projection;
};

struct ObjectData {};

struct InstanceTransform {
    float4x4 model;
};

struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

struct VsIn {
    [[vk::location(0)]] float3 pos: POSITION;
    [[vk::location(3)]] float2 uv: TEXCOORD0;
    uint instance: SV_InstanceID;
};

[[vk::binding(0, 0)]] ConstantBuffer<PassData> pass_data;
[[vk::binding(0, 2)]] ConstantBuffer<ObjectData> object_data;
[[vk::binding(0, 3)]] StructuredBuffer<InstanceTransform> transforms;

VsOut main(VsIn vsin) {
    VsOut vsout;
    
    vsout.position = mul(pass_data.projection, mul(pass_data.view, mul(transforms[vsin.instance].model, float4(vsin.pos, 1.0))));
    vsout.uv = vsin.uv;

    return vsout;
}