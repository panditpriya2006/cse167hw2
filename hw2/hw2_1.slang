[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }
    float2 p = float2(ipixel) + float2(0.5, 0.5);
float2 v0 = sp1 - sp0;
float2 v1 = sp2 - sp0;
float2 v2 = p - sp0;
float denom = v0.x * v1.y - v1.x * v0.y;
if (denom == 0.0) {
    return;
}
float w1 = (v2.x * v1.y - v1.x * v2.y) / denom;
float w2 = (v0.x * v2.y - v2.x * v0.y) / denom;
float w0 = 1.0 - w1 - w2;
if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) {
    image[ipixel] = float4(color, 1.0);
}

}
