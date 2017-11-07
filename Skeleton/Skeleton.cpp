//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

int const R = 2, r = 1;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	uniform vec2 texCursor;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texCoord);
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

// 3D point in Cartesian coordinates or RGB color
struct vec3 {
	float v[3];

	vec3(float x = 0, float y = 0, float z = 0) { v[0] = x; v[1] = y; v[2] = z; }
	vec3(const vec4& hom) { v[0] = hom.v[0] / hom.v[3]; v[1] = hom.v[1] / hom.v[3]; v[2] = hom.v[2] / hom.v[3]; }
	vec3 operator*(float s) {
		vec3 res(v[0] * s, v[1] * s, v[2] * s);
		return res;
	}

	void operator+=(const vec3& rhs) {
		v[0] += rhs.v[0];
		v[1] += rhs.v[1];
		v[2] += rhs.v[2];
	}
};

// 2D point in Cartesian coordinates
struct vec2 {
	float v[2];

	vec2(float x = 0, float y = 0) { v[0] = x; v[1] = y; }
	vec2(const vec4& hom) { v[0] = hom.v[0] / hom.v[3]; v[1] = hom.v[1] / hom.v[3]; }
	vec2 operator-(vec2& right) {
		vec2 res(v[0] - right.v[0], v[1] - right.v[1]);
		return res;
	}
	float Length() { return sqrtf(v[0] * v[0] + v[1] * v[1]); }
};


// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class BezierCurve {
	std::vector<vec3> cps;	// control points

	float B(int i, float t) {
		int n = cps.size() - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
public:
	void AddControlPoint(vec3 cp) { cps.push_back(cp); }

	vec3 r(float t) {
		vec3 rr(0, 0);
		for (int i = 0; i < cps.size(); i++) rr += cps[i] * B(i, t);
		return rr;
	}
};

class TextureBackground {
	unsigned int vao, vbo[2], textureId;	// vertex array object id and texture id
	vec2 vertices[4], uvs[4];
public:
	TextureBackground() {
		vertices[0] = vec2(-10, -10); uvs[0] = vec2(0, 0);
		vertices[1] = vec2(10, -10);  uvs[1] = vec2(1, 0);
		vertices[2] = vec2(10, 10);   uvs[2] = vec2(1, 1);
		vertices[3] = vec2(-10, 10);  uvs[3] = vec2(0, 1);
	}
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		glGenBuffers(2, vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which is not modified 
																					   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
																			   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		int width = 128, height = 128;
		vec3 * image = new vec3[width * height];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float gauss_curvature = (1 / r)*(cosf((float)(2 * 3.14)*x / width) / R + r*cosf((float)(2 * 3.14)*x / width));
				if(gauss_curvature > 0)
					image[y * width + x] = vec3(gauss_curvature/1.5, 0, 0);
				else
					image[y * width + x] = vec3(0, -gauss_curvature/1.5, 0);
			}
		}
		// Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source

		mat4 MVPTransform = camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
public:
	Triangle() {
		Animate(0);
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -8, -8, -6, 10, 8, -2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			         sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0); 
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			                  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

		// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 Mscale(sx, 0, 0, 0,
			        0, sy, 0, 0,
			        0, 0, 0, 0,
			        0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1,   0,  0, 0,
			            0,   1,  0, 0,
			            0,   0,  0, 0,
			          wTx, wTy,  0, 1); // model matrix

		mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[8];	// interleaved data of coordinates and colors
	float  cps[26];			// control points' coordinates
	float  test[1000];
	int    nVertices;       // number of vertices
	int	   nCps;			// number of control points
	int    nTest;
	unsigned int textureId;	// 

	void CreateBezierCurve(BezierCurve curve) {
		vec3 tmp;
		for (float t = 0.0; t <= 1.0; t += 0.02) {
			tmp = curve.r(t);
			test[nTest * 2 + 0] = tmp.v[0];
			test[nTest * 2 + 1] = tmp.v[1];
			
			printf("%d. v: %f, u: %f \t", nTest, test[nTest * 2 + 0], test[nTest * 2 + 1]);
			printf("[0,1] (u,v): (%f,%f) \n", (test[nTest * 2 + 0]+10)/20, (test[nTest * 2 + 1]+10)/20);

			nTest++;
		}
	}

	void DoAlgorithm(BezierCurve curve) {
		
	}
	void Perturb() {

	}
	void Lenght() {

	}
public:
	LineStrip() {
		nVertices = 0;
		nCps = 0;
		nTest = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));


		int width = 128, height = 128;
		vec3 * image = new vec3[width * height];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				image[y * width + x] = vec3(1, 1, 1);
			}
		}
		// Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void AddPoint(float cX, float cY) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		if (nVertices >= 3) {
			nVertices = 0;
			nCps = 0;
			nTest = 0;
		}

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[2 * nVertices]     = wVertex.v[0];
		vertexData[2 * nVertices + 1] = wVertex.v[1];
		nVertices++;

		if (nVertices == 2) {
			BezierCurve curveAB;
			curveAB.AddControlPoint(vec3(vertexData[0 * 2 + 0], vertexData[0 * 2 + 1]));
			curveAB.AddControlPoint(vec3((float)((vertexData[0 * 2 + 0] + vertexData[1 * 2 + 0])/2 + 2), (float)((vertexData[0 * 2 + 1] + vertexData[1 * 2 + 1])/2)));
			curveAB.AddControlPoint(vec3(vertexData[1 * 2 + 0], vertexData[1 * 2 + 1]));

			printf("CP_1 \t v: %f, u: %f \n", vertexData[0 * 2 + 0], vertexData[0 * 2 + 1]);
			printf("CP_2 \t v: %f, u: %f \n", (float)((vertexData[0 * 2 + 0] + vertexData[1 * 2 + 0]) / 2) + 2, (float)((vertexData[0 * 2 + 1] + vertexData[1 * 2 + 1]) / 2) + 2);
			printf("CP_3 \t v: %f, u: %f \n\n", vertexData[1 * 2 + 0], vertexData[1 * 2 + 1]);

			CreateBezierCurve(curveAB);
			DoAlgorithm(curveAB);
		}
		else if (nVertices == 3) {
			BezierCurve curveBC;
			curveBC.AddControlPoint(vec3(vertexData[1 * 2 + 0], vertexData[1 * 2 + 1]));
			curveBC.AddControlPoint(vec3((float)((vertexData[1 * 2 + 0] + vertexData[2 * 2 + 0]) / 2) + 2, (float)((vertexData[1 * 2 + 1] + vertexData[2 * 2 + 1]) / 2) + 2));
			curveBC.AddControlPoint(vec3(vertexData[2 * 2 + 0], vertexData[2 * 2 + 1]));

			printf("CP_1 \t v: %f, u: %f \n", vertexData[1 * 2 + 0], vertexData[1 * 2 + 1]);
			printf("CP_2 \t v: %f, u: %f \n", (float)((vertexData[1 * 2 + 0] + vertexData[2 * 2 + 0]) / 2) + 2, (float)((vertexData[1 * 2 + 1] + vertexData[2 * 2 + 1]) / 2) + 2);
			printf("CP_3 \t v: %f, u: %f \n\n", vertexData[2 * 2 + 0], vertexData[2 * 2 + 1]);

			CreateBezierCurve(curveBC);

			BezierCurve curveCA;
			curveCA.AddControlPoint(vec3(vertexData[2 * 2 + 0], vertexData[2 * 2 + 1]));
			curveCA.AddControlPoint(vec3((float)((vertexData[2 * 2 + 0] + vertexData[0 * 2 + 0]) / 2) + 2, (float)((vertexData[2 * 2 + 1] + vertexData[0 * 2 + 1]) / 2) + 2));
			curveCA.AddControlPoint(vec3(vertexData[0 * 2 + 0], vertexData[0 * 2 + 1]));

			printf("CP_1 \t v: %f, u: %f \n", vertexData[2 * 2 + 0], vertexData[2 * 2 + 1]);
			printf("CP_2 \t v: %f, u: %f \n", (float)((vertexData[2 * 2 + 0] + vertexData[0 * 2 + 0]) / 2) + 2, (float)((vertexData[2 * 2 + 1] + vertexData[0 * 2 + 1]) / 2) + 2);
			printf("CP_3 \t v: %f, u: %f \n\n", vertexData[0 * 2 + 0], vertexData[0 * 2 + 1]);

			CreateBezierCurve(curveCA);
		}
		
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nTest * 2 * sizeof(float), test, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			location = glGetUniformLocation(shaderProgram, "textureUnit");
			if (location >= 0) {
				glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
			}

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nTest);
		}
	}
};

// The virtual world: collection of two objects
Triangle triangle;
LineStrip lineStrip;
TextureBackground background;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	//lineStrip.Create();
	//triangle.Create();

	background.Create();
	lineStrip.Create();


	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	//triangle.Draw();
	//lineStrip.Draw();

	background.Draw();
	lineStrip.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;		// pont transzformálása normalizált koordináta rendszerbe
		lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw

		/*printf("v: %f, u: %f" "\n", ((float)pX/windowWidth), ((float)pY/windowHeight));*/
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	/*
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	//triangle.Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene
	*/
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

