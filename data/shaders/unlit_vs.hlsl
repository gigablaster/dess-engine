struct PassData {
    float4x4 view;
    float4x4 projection;
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
};

[[vk::binding(0, 0)]] ConstantBuffer<PassData> pass_data;
[[vk::binding(0, 1)]] ConstantBuffer<InstanceTransform> transform;

VsOut main(VsIn vsin) {
    VsOut vsout;
    
    float4x4 mvp = transform.model * pass_data.view * pass_data.projection;
    vsout.position = mul(mvp, float4(vsin.pos, 0.0));
    vsout.uv = vsin.uv;

    return vsout;
}