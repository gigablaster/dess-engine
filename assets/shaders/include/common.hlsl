struct PassData {
    float4x4 view;
    float4x4 projection;
    float3 eye_position;
};

struct InstanceTransform {
    float4x4 model;
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

