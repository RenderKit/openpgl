#include <metal_stdlib>
using namespace metal;

namespace colormap {
namespace IDL {
namespace CB_PuRd {

float colormap_red(float x) {
	if (x < 25.0 / 254.0) {
		return -1.41567647058826E+02 * x + 2.46492647058824E+02;
	} else if (x < 0.3715468440779919) {
		return (1.64817020395145E+02 * x - 2.01032852327719E+02) * x + 2.52820173539371E+02;
	} else if (x < 0.6232413065898157) {
		return (((2.61012828741073E+04 * x - 5.18905872811356E+04) * x + 3.78968358931486E+04) * x - 1.19124127524292E+04) * x + 1.55945779375675E+03;
	} else if (x < 0.7481208809057023) {
		return -2.02469919786095E+02 * x + 3.57739416221033E+02;
	} else {
		return -4.08324020737294E+02 * x + 5.11743167562695E+02;
	}
}

float colormap_green(float x) {
	if (x < 0.1303350956955242) {
		return -1.59734759358287E+02 * x + 2.44376470588235E+02;
	} else if (x < 0.6227215280200861) {
		return (((1.21347373400442E+03 * x - 2.42854832541048E+03) * x + 1.42039752537243E+03) * x - 6.27806679597789E+02) * x + 2.86280758506240E+02;
	} else {
		return (1.61877993987291E+02 * x - 4.06294499392671E+02) * x + 2.32401278080262E+02;
	}
}

float colormap_blue(float x) {
	if (x < 0.7508644163608551) {
		return ((((2.96852143551409E+03 * x - 6.12155011029541E+03) * x + 4.21719423212110E+03) * x - 1.29520280960574E+03) * x + 2.78723913454450E+01) * x + 2.47133504519275E+02;
	} else {
		return ((-6.55064010825706E+02 * x + 1.23635622822904E+03) * x - 8.68481725874416E+02) * x + 3.18158180572088E+02;
	}
}

float4 colormap(float x) {
	float r = clamp(colormap_red(x) / 255.0, 0.0, 1.0);
	float g = clamp(colormap_green(x) / 255.0, 0.0, 1.0);
	float b = clamp(colormap_blue(x) / 255.0, 0.0, 1.0);
	return float4(r, g, b, 1.0);
}

} // namespace CB_PuRd
} // namespace IDL
} // namespace colormap
