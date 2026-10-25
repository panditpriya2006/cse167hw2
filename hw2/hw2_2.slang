[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1

    // Pixel sample point is at center (x + 0.5, y + 0.5)
    float2 p = float2(ipixel) + float2(0.5, 0.5);

    // 1. Edge test (Eq. 4)
    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float d0 = dot(p - sp0, n01);
    float d1 = dot(p - sp1, n12);
    float d2 = dot(p - sp2, n20);

    bool inside = (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || 
                  (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0);
    if (!inside) {
        return;
    }

    // 2. Projected 2D barycentric coordinates (Eq. 6)
    float2 e0 = sp1 - sp0;
    float2 e1 = sp2 - sp0;
    float total_area = e0.x * e1.y - e0.y * e1.x;
    if (abs(total_area) < 1e-8) {
        return;
    }

    float2 v21 = sp2 - sp1;
    float2 vp1 = p - sp1;
    float b0_prime = (v21.x * vp1.y - v21.y * vp1.x) / total_area;

    float2 v02 = sp0 - sp2;
    float2 vp2 = p - sp2;
    float b1_prime = (v02.x * vp2.y - v02.y * vp2.x) / total_area;

    float b2_prime = 1.0 - b0_prime - b1_prime;

    // 3. Perspective-correct interpolation using inverse depth (Eq. 14 & 15)
    // Note: inv_depth = 1 / d = 1 / (-z) = -1 / z
    // w_i = b'_i * inv_depth_i
    float w0 = b0_prime * inv_depth0;
    float w1 = b1_prime * inv_depth1;
    float w2 = b2_prime * inv_depth2;
    float sum_w = w0 + w1 + w2;

    if (sum_w <= 0.0) {
        return;
    }

    float b0 = w0 / sum_w;
    float b1 = w1 / sum_w;
    float b2 = w2 / sum_w;

    // Depth d = -z = 1.0 / sum_w
    float depth = 1.0 / sum_w;

    // 4. Depth test and write
    if (depth < depth_image[ipixel]) {
        depth_image[ipixel] = depth;
        float3 final_color = b0 * c0 + b1 * c1 + b2 * c2;
        color_image[ipixel] = float4(final_color, 1.0);
    }

    //color_image[ipixel] = float4(float3(0.0), 1.0);
}
