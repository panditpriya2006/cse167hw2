
float triangle_area(float2 p0, float2 p1, float2 p2)
{
    return (p1.x - p0.x) * (p2.y - p0.y)
         - (p1.y - p0.y) * (p2.x - p0.x);
}

// Compute screen-space barycentric coordinates.
// Returns false for degenerate triangles or pixels outside the triangle.
bool get_barycentric(
    float2 p,
    float2 p0,
    float2 p1,
    float2 p2,
    out float3 bary)
{
    float area = triangle_area(p0, p1, p2);

    bary = float3(0.0, 0.0, 0.0);

    if (abs(area) < 1e-8)
        return false;

    float w0 = triangle_area(p, p1, p2) / area;
    float w1 = triangle_area(p, p2, p0) / area;
    float w2 = triangle_area(p, p0, p1) / area;

    bary = float3(w0, w1, w2);

    return w0 >= -1e-6 && w1 >= -1e-6 && w2 >= -1e-6;
}

void get_triangle(
    uint triangle_id,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> projected_vertices,
    out float3 p0,
    out float3 p1,
    out float3 p2)
{
    uint3 face = faces[triangle_id];

    p0 = projected_vertices[face.x];
    p1 = projected_vertices[face.y];
    p2 = projected_vertices[face.z];
}

void get_pixel_bounds(
    float3 p0,
    float3 p1,
    float3 p2,
    uint width,
    uint height,
    out int min_x,
    out int min_y,
    out int max_x,
    out int max_y)
{
    min_x = max(0, (int)floor(min(p0.x, min(p1.x, p2.x))));
    min_y = max(0, (int)floor(min(p0.y, min(p1.y, p2.y))));
    max_x = min((int)width - 1,
                (int)ceil(max(p0.x, max(p1.x, p2.x))));
    max_y = min((int)height - 1,
                (int)ceil(max(p0.y, max(p1.y, p2.y))));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(
    uint tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    RWStructuredBuffer<uint> depth_buffer,
    RWStructuredBuffer<uint> triangle_id_buffer)
{
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;

    float4 p = mul(MVP, float4(vertices[i], 1.0));

    projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
    uint tid : SV_DispatchThreadID,
    uniform uint num_triangles,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> projected_vertices,
    RWStructuredBuffer<uint> depth_buffer)
{
    if (tid >= num_triangles)
        return;

    float3 p0, p1, p2;
    get_triangle(tid, faces, projected_vertices, p0, p1, p2);

    // Reject degenerate projected triangles.
    float area = triangle_area(p0.xy, p1.xy, p2.xy);
    if (abs(area) < 1e-8)
        return;

    int min_x, min_y, max_x, max_y;
    get_pixel_bounds(
        p0, p1, p2, width, height,
        min_x, min_y, max_x, max_y
    );

    if (min_x > max_x || min_y > max_y)
        return;

    for (int y = min_y; y <= max_y; ++y)
    {
        for (int x = min_x; x <= max_x; ++x)
        {
            // Test the pixel center.
            float2 pixel_center = float2(x + 0.5, y + 0.5);
            float3 bary;

            if (!get_barycentric(
                    pixel_center, p0.xy, p1.xy, p2.xy, bary))
                continue;

            // Interpolate inverse depth.
            float d = dot(bary, float3(p0.z, p1.z, p2.z));

            // The unsigned bit representation preserves ordering
            // for positive finite floating-point depths.
            if (d > 0.0 && isfinite(d))
            {
                uint pixel = (uint)y * width + (uint)x;
                InterlockedMin(depth_buffer[pixel], asuint(d));
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
    uint tid : SV_DispatchThreadID,
    uniform uint num_triangles,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> projected_vertices,
    RWStructuredBuffer<uint> depth_buffer,
    RWStructuredBuffer<uint> triangle_id_buffer)
{
    if (tid >= num_triangles)
        return;

    float3 p0, p1, p2;
    get_triangle(tid, faces, projected_vertices, p0, p1, p2);

    // Use the same degenerate-triangle test as the depth pass.
    float area = triangle_area(p0.xy, p1.xy, p2.xy);
    if (abs(area) < 1e-8)
        return;

    int min_x, min_y, max_x, max_y;
    get_pixel_bounds(
        p0, p1, p2, width, height,
        min_x, min_y, max_x, max_y
    );

    if (min_x > max_x || min_y > max_y)
        return;

    for (int y = min_y; y <= max_y; ++y)
    {
        for (int x = min_x; x <= max_x; ++x)
        {
            float2 pixel_center = float2(x + 0.5, y + 0.5);
            float3 bary;

            if (!get_barycentric(
                    pixel_center, p0.xy, p1.xy, p2.xy, bary))
                continue;

            float d = dot(bary, float3(p0.z, p1.z, p2.z));

            if (d > 0.0 && isfinite(d))
            {
                uint pixel = (uint)y * width + (uint)x;
                uint depth_bits = asuint(d);

                if (depth_bits == depth_buffer[pixel])
                {
                    // Resolve equal-depth collisions deterministically.
                    InterlockedMin(triangle_id_buffer[pixel], tid);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
    uint2 tid : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> vertex_colors,
    StructuredBuffer<float3> projected_vertices,
    StructuredBuffer<uint> triangle_id_buffer,
    RWTexture2D<float4> image)
{
    if (tid.x >= width || tid.y >= height)
        return;

    uint pixel = tid.y * width + tid.x;
    uint triangle_id = triangle_id_buffer[pixel];

    // The pixel is not covered by any triangle.
    if (triangle_id == 0xffffffff)
        return;

    uint3 face = faces[triangle_id];

    float3 p0 = projected_vertices[face.x];
    float3 p1 = projected_vertices[face.y];
    float3 p2 = projected_vertices[face.z];

    // Reject degenerate triangles consistently.
    float area = triangle_area(p0.xy, p1.xy, p2.xy);
    if (abs(area) < 1e-8)
        return;

    float2 pixel_center = float2(tid.x + 0.5, tid.y + 0.5);
    float3 bary;

    if (!get_barycentric(
            pixel_center, p0.xy, p1.xy, p2.xy, bary))
        return;

    float3 c0 = vertex_colors[face.x];
    float3 c1 = vertex_colors[face.y];
    float3 c2 = vertex_colors[face.z];

    float3 color = bary.x * c0 + bary.y * c1 + bary.z * c2;

    image[tid] = float4(color, 1.0);
}
