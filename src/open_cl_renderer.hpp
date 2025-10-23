#pragma once
//#define PTX
#define LOG
#define UTILITIES_FILE
#include"..\third_party\OpenCL-Wrapper\opencl.hpp"


#include "utils/va_grid.hpp"
#include "utils/thread_pool.hpp"

#include "config.hpp"



const std::string juliaKernelSource = R"CLC(
__constant double julia_r = -0.8;
__constant double julia_i = 0.156;
__constant uint samples_count = 9;

__constant double2 anti_aliasing_offsets[16] = {
    (double2)( 0.00,  0.00),
    (double2)(-0.15,  0.15),
    (double2)( 0.15, -0.15),
    (double2)( 0.15,  0.15),
    (double2)(-0.15, -0.15),
    (double2)(-0.15,  0.00),
    (double2)( 0.00, -0.15),
    (double2)( 0.15,  0.00),
    (double2)( 0.00,  0.15),
};

inline double julia_iter(double2 z, uint maxIterations) {
    uint i = 0;
    double mod = dot(z, z);

    while (mod < 4.0 && i < maxIterations) {
        double2 z2 = (double2)(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
        z = z2 + (double2)(julia_r, julia_i);
        mod = dot(z, z);
        ++i;
    }

    // classic smooth iteration count (guarded)
    // nu = i + 1 - log(log(|z|)) / log(2)
    if (mod <= 0.0) {
        return (float)i;
    }
    double log_mod = log(mod);
    if (log_mod <= 0.0) {
        return (float)i;
    }
    double nu = (double)i + 1.0f - (double)(log(log_mod) / M_LN2); // M_LN2 = ln(2), some OpenCL compilers don't provide it; see alternative below
// float nu = (float)i + 1.0f - (float)(log(log_mod) / log(2.0));
    return nu;
}

inline float4 interpolate_color(float value, __global const float4* palette, uint palette_size) {
    if (value <= palette[0].w) return palette[0];
    if (value >= palette[palette_size - 1].w) return palette[palette_size - 1];

    for (uint i = 1; i < palette_size; ++i) {
        if (palette[i].w > value) {
            float range = palette[i].w - palette[i - 1].w;
            float pos   = value - palette[i - 1].w;
            float ratio = pos / range;
            return (1.0f - ratio) * palette[i - 1] + ratio * palette[i];
        }
    }
    return (float4)(255.0f, 255.0f, 255.0f, 1.0f); // fallback white
}


__kernel void generate_julia(
    __global uchar4* result,
    __global const float4* palette,
    const uint palette_size,
    const int width,
    const int height,
    const uint maxIterations,
    const double render_zoom,
    const double center_x,
    const double center_y
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const double2 win_center = (double2)(0.5 * (double)width, 0.5 * (double)height);
    const double2 center = (double2)(center_x, center_y);
    const double2 pixel = (double2)(x, y);
    const double invMaxIter   = 1.0 / (double)maxIterations;
    float4 color_accum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint i = 0; i < samples_count; ++i) {
        const double2 sample_pos  = ( (pixel + anti_aliasing_offsets[i] ) - win_center) / render_zoom + center;

        const double iter_ratio = julia_iter(sample_pos, maxIterations) * invMaxIter;

        color_accum += interpolate_color((float)iter_ratio, palette, palette_size);
    }

    float4 final_color = color_accum / (float)samples_count;
    result[y * width + x] = (uchar4)(
        clamp(final_color.x, 0.0f, 255.0f),
        clamp(final_color.y, 0.0f, 255.0f),
        clamp(final_color.z, 0.0f, 255.0f),
        255
    );

}
)CLC";



    static const cl_float4 static_palette[] = {
        {  25.0f,  24.0f,  23.0f,  0.0f   },
        { 120.0f,  90.0f,  70.0f,  0.03f  },
        { 130.0f,  24.0f,  23.0f,  0.05f  },
        { 179.0f, 100.0f, 250.0f,  0.15f  },
        { 250.0f, 179.0f, 100.0f,  0.25f  },
        {  43.0f,  65.0f,  98.0f,  0.5f   },
        { 100.0f, 179.0f, 250.0f,  0.65f  },
        { 110.0f,  79.0f,  11.0f,  0.75f  },
        {  79.0f,  11.0f, 110.0f,  0.80f  },
        {  11.0f, 110.0f,  79.0f,  0.85f  },
        { 150.0f, 110.0f,  79.0f,  0.95f  },
        { 255.0f, 255.0f, 255.0f,  1.0f   }
    };



constexpr uint palette_size = sizeof(static_palette) / sizeof(static_palette[0]);





template<typename TFloatType>
struct RenderState
{
    VertexArrayGrid         grid;
    TFloatType              zoom = 1.0;
    sf::Vector2<TFloatType> offset = { 0.0, 0.0 };

    RenderState(uint32_t width, uint32_t height)
        : grid(width, height)
    {}
};


Memory<const cl_float4> create_palette_memory(Device& device) {
    return Memory<const cl_float4>(
        device,
        palette_size,
        1u,
        const_cast<cl_float4*>(static_palette),
        true,   // allocate_device
        false   // allow_zero_copy
        );
}

template<typename TFloatType>
struct OpenClRenderer
{
    Device device;
    Memory<cl_uchar4> memoryState;
    Memory<const cl_float4> memoryPalette;
    Kernel juliaKernel;


    RenderState<TFloatType> states[2];

    TFloatType              requested_zoom = 1.0;
    sf::Vector2<TFloatType> requested_center = {};

    TFloatType              render_zoom = 1.0;
    sf::Vector2<TFloatType> render_center = {};

    uint32_t state_idx = 0;
    uint32_t texture_idx = 0;

    sf::RenderTexture                       textures[2];

    uint32_t _width;
    uint32_t _height;

    tp::ThreadPool thread_pool;



    float fade_time = Config::fade_time;

    OpenClRenderer(uint32_t width, uint32_t height, TFloatType zoom_)
		: _width(width), _height(height),
        device(select_device_with_most_flops(), juliaKernelSource),
        memoryState(device, width * height),
        memoryPalette(create_palette_memory(device)),
        juliaKernel(device, width * height, "generate_julia", memoryState, memoryPalette, palette_size, width, height, Config::max_iteration),
        thread_pool{ 1 },
        states{ RenderState<TFloatType>(width, height), RenderState<TFloatType>(width, height) }
    {
        requested_zoom = zoom_;

        memoryPalette.write_to_device();

        for (uint32_t i{ 2 }; i--;) {
            textures[i].create(width, height);
            textures[i].setSmooth(true);
        }
    }

    ~OpenClRenderer()
    {
        thread_pool.waitForCompletion();
    }

    void generate()
    {
        thread_pool.addTask([=] {

            render_zoom = requested_zoom;
            render_center = requested_center;

            juliaKernel.set_parameters(6, render_zoom, render_center.x, render_center.y);
            juliaKernel.set_ranges_2d(_width, _height, 8);

            juliaKernel.enqueue_run();

            memoryState.read_from_device();

            auto& grid = states[!state_idx].grid;

            for (int y = 0; y != _height; ++y) {
                int stride = y * _width;
                for (int x = 0; x != _width; ++x) {
                    auto result = memoryState[x + stride];
                    sf::Color color(result.x, result.y, result.z);
                    grid.setCellColor(x, y, color);
                }
            }
        });
    }

    void render(TFloatType zoom, sf::Vector2<TFloatType> center, sf::RenderTarget& target)
    {
        // Check if background rendering is ready
        if (fade_time >= Config::fade_time && thread_pool.isDone())
        {
            // Update state
            auto& state = states[!state_idx];
            state.zoom = render_zoom;
            state.offset = render_center;
            // Swap buffers
            state_idx = !state_idx;
            // Update texture
            textures[!texture_idx].draw(states[state_idx].grid.va);
            textures[!texture_idx].display();
            texture_idx = !texture_idx;
            fade_time = 0.0f;
            // Start next render
            generate();
        }

        requested_center = center;
        requested_zoom = zoom;
        const auto scale = getStateScale(state_idx);
        sf::Sprite sprite_new(textures[texture_idx].getTexture());
        const auto bounds = sprite_new.getGlobalBounds();
        const sf::Vector2f origin = sf::Vector2f{ bounds.width, bounds.height } *0.5f;
        sprite_new.setOrigin(origin);
        sprite_new.setPosition(origin + getStateOffset(state_idx));
        sprite_new.setScale(scale, scale);

        const auto scale_old = getStateScale(!state_idx);
        sf::Sprite sprite_old(textures[!texture_idx].getTexture());
        sprite_old.setOrigin(origin);
        sprite_old.setPosition(origin + getStateOffset(!state_idx));
        sprite_old.setScale(scale_old, scale_old);
        const auto alpha = static_cast<uint8_t>(std::max(0.0f, 1.0f - fade_time / Config::fade_time) * 255.0f);
        sprite_old.setColor(sf::Color{ 255, 255, 255, alpha });

        target.draw(sprite_new);
        target.draw(sprite_old);

        fade_time += 0.016f;
    }

    float getStateScale(uint32_t idx) const
    {
        return static_cast<float>(requested_zoom / states[idx].zoom);
    }

    sf::Vector2<float> getStateOffset(uint32_t idx) const
    {
        return static_cast<sf::Vector2f>((states[idx].offset - requested_center) * states[idx].zoom) * getStateScale(idx);
    }
};