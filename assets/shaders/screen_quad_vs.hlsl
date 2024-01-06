struct VsIn {
    [[vk::location(0)]] float3 position: POSITION;
    [[vk::location(1)]] float2 uv: TEXCOORD0;
};

struct VsOut {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

VsOut main(VsIn vsin) {
    VsOut result;

    result.position = float4(vsin.position, 1.0);
    result.uv = vsin.uv;

    return result;
}