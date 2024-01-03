struct PassData {
    float4x4 view;
    float4x4 projection;
    float3 eye_position;
};

struct ObjectData {};

struct InstanceTransform {
    float4x4 model;
};

struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float2 uv1: TEXCOORD1;
    [[vk::location(2)]] float2 uv2: TEXCOORD2;
    [[vk::location(3)]] float3x3 tangent_basis: TBASIS;
};

struct VsIn {
    [[vk::location(0)]] float3 pos: POSITION;
    [[vk::location(1)]] float3 normal: NORMAL;
    [[vk::location(2)]] float3 tangent: TEXCOORD0;
    [[vk::location(3)]] float2 uv1: TEXCOORD1;
    [[vk::location(4)]] float2 uv2: TEXCOORD2;
    uint id: SV_INSTANCEID;
};

[[vk::binding(0, 0)]] ConstantBuffer<PassData> pass_data;
[[vk::binding(0, 2)]] ConstantBuffer<ObjectData> object_data;
[[vk::binding(0, 3)]] StructuredBuffer<InstanceTransform> transforms;

VsOut main(VsIn vsin) {
    VsOut vsout;

    float4x4 model = transforms[vsin.id].model;
    float4x4 mvp = mul(pass_data.projection, mul(pass_data.view, model));
    float4 pos = mul(mvp, float4(vsin.pos, 1.0));
    float3 binormal = cross(vsin.normal, vsin.tangent);
    float3x3 tbn = float3x3(vsin.tangent, binormal, vsin.normal);
    float3x3 tangent_basis = mul((float3x3)model, transpose(tbn));

    vsout.pos = pos.xyz;
    vsout.position = pos;
    vsout.tangent_basis = tangent_basis;
    vsout.uv1 = vsin.uv1;
    vsout.uv2 = vsin.uv2;

    return vsout;
}