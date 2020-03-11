#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <thread>
#include <atomic>
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

#include <Windows.h>

#define M_PI 3.14159265359f  // pi
#define width 1920  // screenwidth
#define height 1080 // screenheight
#define samps 16384 // samples 

using float3 = concurrency::graphics::float_3;
namespace math = concurrency::fast_math;

#define GPU restrict(amp, cpu)
#define GPU_ONLY restrict(amp)

inline float clamp(float f, float a, float b) GPU {
  using namespace math;
  return fmaxf(a, fminf(f, b));
}
inline float3 clamp_f3(const float3& v, float a, float b) GPU {
  return float3(clamp(v.x, a, b),
                clamp(v.y, a, b),
                clamp(v.z, a, b));
}

inline float3 clamp_f3f3(const float3& v, const float3& a, const float3& b) GPU {
  return float3(clamp(v.x, a.x, b.x),
                clamp(v.y, a.y, b.y),
                clamp(v.z, a.z, b.z));
}

inline float dot(const float3& a, const float3& b) GPU {
  return a.x*b.x +
         a.y*b.y +
         a.z*b.z;
}

inline float3 cross(const float3& a, const float3& b) GPU {
  return float3(a.y*b.z - a.z*b.y,
                a.z*b.x - a.x*b.z,
                a.x*b.y - a.y*b.x);
}

inline float3 normalize(float3 v) GPU_ONLY {
  float invLen = math::rsqrtf(dot(v, v));
  return v * invLen;
}

struct Ray
{
  float3 origin;
  float3 direction;
  constexpr Ray(const float3& o_, const float3& d_) GPU
    : origin(o_)
    , direction(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance(), only DIFF used here

struct Sphere
{
  float rad;            // radius 
  float3 pos, emi, col; // position, emission, colour 
  Refl_t refl;          // reflection type (e.g. diffuse)

  static float intersect(const Sphere& s, const Ray &r) GPU {
    using namespace concurrency;
    using namespace math;
    // ray/sphere intersection
    // returns distance t to intersection point, 0 if no hit  
    // ray equation: p(x,y,z) = ray.origin + t*ray.direction
    // general sphere equation: x^2 + y^2 + z^2 = rad^2 
    // classic quadratic equation of form ax^2 + bx + c = 0 
    // solution x = (-b +- sqrt(b*b - 4ac)) / 2a
    // solve t^2*ray.direction*ray.direction + 2*t*(orig-p)*ray.direction + (origin-p)*(orig-p) - rad*rad = 0 
    // more details in "Realistic Ray Tracing" book by P. Shirley or Scratchapixel.com

    float3 op = s.pos - r.origin;    // distance from ray.origin to center sphere 
    float t, epsilon = 1e-6f;  // epsilon required to prevent floating point precision artefacts
    float b = dot(op, r.direction);    // b in quadratic equation
    float disc = b*b - dot(op, op) + s.rad*s.rad;  // discriminant quadratic equation
    if (disc < 0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
    else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
    return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0); // pick closest point in front of ray origin
  }
};

inline bool intersect_scene(const Ray &r, float &t, int &id, const concurrency::array_view<Sphere, 1>& spheres) GPU {
  float d = 0;
  float inf = 1e20f;
  t = inf;// t is distance to closest intersection, initialise t to a huge number outside scene
  for (int i = 0; i < 9; i++) { // test all scene objects for intersection
    if ((d = Sphere::intersect(spheres[i], r)) && d < t) {  // if newly computed intersection distance d is smaller than current closest intersection distance
      t = d;  // keep track of distance along ray to closest intersection point 
      id = i; // and closest intersected object
    }
  }
  return t < inf; // returns true if an intersection with the scene occurred, false when no hit
}

// random number generator from https://github.com/gz/rust-raytracer

static float getrandom(unsigned int *seed0, unsigned int *seed1) GPU {
  *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
  *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

  unsigned int ires = ((*seed0) << 16) + (*seed1);

  // Convert to float
  union
  {
    float f;
    unsigned int ui;
  } res;

  res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

  return (res.f - 2.f) / 2.f;
}

// radiance function, the meat of path tracing 
// solves the rendering equation: 
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point, 
// multiplied by reflectance function of material (BRDF) and cosine incident angle 
float3 radiance(Ray &r, unsigned int *s1, unsigned int *s2, const concurrency::array_view<Sphere, 1>& spheres) GPU_ONLY // returns ray color
{
  using namespace concurrency::fast_math;

  float3 accucolor = float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
  float3 mask = float3(1.0f, 1.0f, 1.0f);

  // ray bounce loop (no Russian Roulette used) 
  for (int bounces = 0; bounces < 4; bounces++) {  // iteration up to 4 bounces (replaces recursion in CPU code)

    float t;           // distance to closest intersection 
    int id = 0;        // index of closest intersected sphere 

               // test ray for intersection with scene
    if (!intersect_scene(r, t, id, spheres))
      return float3(0.0f, 0.0f, 0.0f); // if miss, return black

                          // else, we've got a hit!
                          // compute hitpoint and normal
    const Sphere &obj = spheres[id];  // hitobject
    float3 x = r.origin + r.direction*t;          // hitpoint 
    float3 n = normalize(x - obj.pos);    // normal
    float3 nl = dot(n, r.direction) < 0 ? n : n * -1; // front facing normal

                          // add emission of current sphere to accumulated colour
                          // (first term in rendering equation sum) 
    accucolor += mask * obj.emi;

    // all spheres in the scene are diffuse
    // diffuse material reflects light uniformly in all directions
    // generate new diffuse ray:
    // origin = hitpoint of previous ray in path
    // random direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)

    // create 2 random numbers
    float r1 = 2 * M_PI * getrandom(s1, s2); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
    float r2 = getrandom(s1, s2);  // pick random number for elevation
    float r2s = sqrtf(r2);

    // compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction 
    // first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
    float3 w = nl;
    float3 u = normalize(cross((fabs(w.x) > .1 ? float3(0, 1, 0) : float3(1, 0, 0)), w));
    float3 v = cross(w, u);

    // compute random ray direction on hemisphere using polar coordinates
    // cosine weighted importance sampling (favours ray directions closer to normal direction)
    float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

    // new ray origin is intersection point of previous ray with scene
    r.origin = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
    r.direction = d;

    mask *= obj.col;    // multiply with colour of object       
    mask *= dot(d, nl);  // weigh light contribution using cosine of angle between incident light and normal
    mask *= 2;          // fudge factor
  }

  return accucolor;
}


inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction
static std::vector<COLORREF> vec = std::vector<COLORREF>(width*height);
std::atomic<bool> stopDraw = false;
void drawToScreen(const std::vector<float3>& buf, HDC hdc) {
  size_t xy = 0;

  for (auto& i = buf.rbegin(); i != buf.rend(); i++, xy++)
    vec[xy] = RGB(toInt(i->r), toInt(i->g), toInt(i->b));

  HBITMAP map = CreateBitmap(width, // width. 512 in my case
                 height, // height
                 1, // Color Planes, unfortanutelly don't know what is it actually. Let it be 1
                 8 * 4, // Size of memory for one pixel in bits (in win32 4 bytes = 4*8 bits)
                 (void*)vec.data()); // pointer to array
                       // Temp HDC to copy picture
  HDC src = CreateCompatibleDC(hdc); // hdc - Device context for window, I've got earlier with GetDC(hWnd) or GetDC(NULL);
  SelectObject(src, map); // Inserting picture into our temp HDC
              // Copy image from temp HDC to window
  BitBlt(hdc, // Destination
       0,  // x and
       0,  // y - upper-left corner of place, where we'd like to copy
       width, // width of the region
       height, // height
       src, // source
       0,   // x and
       0,   // y of upper left corner  of part of the source, from where we'd like to copy
       SRCCOPY); // Defined DWORD to juct copy pixels. Watch more on msdn;

  DeleteDC(src); // Deleting temp HDC
}


int main() {
  // SCENE
  // 9 spheres forming a Cornell box
  // small enough to be in constant GPU memory
  // { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
  Sphere spheres_[] = {
    { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
    { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right 
    { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
    { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
    { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
    { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
    { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 1
    { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 2
    { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
  };

  const concurrency::array_view<Sphere, 1> spheres(spheres_);

  // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
  concurrency::array<float3, 1> output_raw(width*height), output_tmp(width*height);
  std::vector<float3> result(width*height);
  concurrency::array_view<float3, 1> result_av(result);
  concurrency::array<unsigned int, 1> seed(2 * width*height);

  printf("Start rendering...\n");
  HWND window; HDC dc; window = GetConsoleWindow(); dc = GetDC(window);
  printf("hwnd: %p, hdc: %p\n", window, dc);
// schedule threads on device and launch CUDA kernel from host
  std::thread th([&] { while (!stopDraw) { drawToScreen(result, dc); std::this_thread::yield(); } });
  for (int s = 0; s < samps; s++) {
    concurrency::parallel_for_each(output_raw.extent, [=, &output_raw, &output_tmp, &seed](concurrency::index<1> idx) GPU_ONLY{
      int x = idx[0] % width;
      int y = idx[0] / width;

      unsigned int& s1 = seed[idx[0] * 2];
      unsigned int& s2 = seed[idx[0] * 2 + 1];

      if (s == 0) {
        s1 = x;
        s2 = y;
      }

      // generate ray directed at lower left corner of the screen
      // compute directions for all other rays by adding cx and cy increments in x and y direction
      Ray cam(float3(50, 52, 295.6f), normalize(float3(0, -0.042612f, -1))); // first hardcoded camera ray(origin, direction) 
      float3 cx = float3(width * .5135f / height, 0.0f, 0.0f); // ray direction offset in x direction
      float3 cy = normalize(cross(cx, cam.direction)) * .5135f; // ray direction offset in y direction (.5135 is field of view angle)
      float3 r; // r is final pixel color       

      r = float3(0.0f); // reset r to zero for every pixel 

        // samples per pixel

                         // compute primary ray direction
      float3 d = cam.direction + cx*((.25f + x) / width - .5f) + cy*((.25f + y) / height - .5f);

      // create primary ray, add incoming radiance to pixelcolor
      r = radiance(Ray(cam.origin + d * 40, normalize(d)), &s1, &s2, spheres)*(1.f / (s + 1));
         // Camera rays are pushed ^^^^^ forward to start in interior 

        // write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
      output_raw[idx] = (output_tmp[idx] * s / (s + 1.f)) + float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
    });
    concurrency::copy(output_raw, output_tmp);
    concurrency::copy(output_tmp, result_av);
  }
  result_av.synchronize();
  stopDraw = true;
  th.join();
  ReleaseDC(window, dc);
  printf("Done!\n");

  // Write image to PPM file, a very simple image file format
  FILE *f = fopen("smallptcuda.ppm", "w");
  fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
  for (auto& i = result.rbegin(); i != result.rend(); i++)  // loop over pixels, write RGB values
    fprintf(f, "%d %d %d ", toInt(i->x),
        toInt(i->y),
        toInt(i->z));

  printf("Saved image to 'smallptcuda.ppm'\n");

  system("PAUSE");
}