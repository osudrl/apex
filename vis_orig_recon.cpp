#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <stdexcept>

using namespace std;

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
bool ispaused = true;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;
int data_ind = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    if (act == GLFW_RELEASE) {
        return;
    } else if (act == GLFW_PRESS) {
        if (key == GLFW_KEY_P && mods == 0) {
            printf("attaching camera to pelvis\n");
            cam.type = mjCAMERA_TRACKING;
            cam.trackbodyid = 1;
            cam.fixedcamid = -1;
            mjv_moveCamera(m, mjMOUSE_ZOOM, 0.0, -0.05*8, &scn, &cam);
            mjv_moveCamera(m, act, 0, -.15, &scn, &cam);
        }
        // control keys
        if (mods == GLFW_MOD_CONTROL) {
            if (key == GLFW_KEY_Q) {
                glfwSetWindowShouldClose(window, true);
            }
        }
        
        if (key == GLFW_KEY_BACKSPACE) {
            // backspace: reset to beginning of data
            data_ind = 0;
        } else if (key == GLFW_KEY_SPACE) {
            // space: toggle pause
            ispaused = !ispaused;
        } else if (key == GLFW_KEY_ESCAPE) {
            cam.type = mjCAMERA_FREE;
        }
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

int main (int argc, const char** argv0) {
    if (argc != 2) {
        throw invalid_argument("Requires single data file");
    }

    // Load csv data
    ifstream datafile(argv0[1]);
    string line;
    vector<vector<double>> orig_states;
    vector<vector<double>> recon_states;
    if (!datafile.is_open()) {
        perror("error while opening file");
        return 1;
    }
    while (getline(datafile, line)) {
        string val;
        vector<double> orig_row;
        stringstream s (line);
        for (int i = 0; i < 35; i++) {
            getline(s, val, ',');
            orig_row.push_back(stod(val));
        }
        orig_states.push_back(orig_row);
        vector<double> recon_row;
        for (int i = 0; i < 35; i++) {
            getline(s, val, ',');
            recon_row.push_back(stod(val));
        }
        recon_states.push_back(recon_row);
    }
    datafile.close();
    int traj_len = orig_states.size();
    printf("Traj len: %i %i\n", orig_states.size(), recon_states.size());
    printf("Data len: %i %i\n", orig_states[0].size(), recon_states[0].size());

    // Activate MuJoCo
    const char* key_buf = getenv("MUJOCO_KEY_PATH");
    mj_activate(key_buf);

    // Load and compile model
    char error[1000] = "Could not load binary model";
    m = mj_loadXML("./cassie/cassiemujoco/cassie_double.xml", 0, error, 1000);
    if (!m) {
        mju_error_s("Load model error: %s", error);
    }

    // Made mjData
    d = mj_makeData(m);
    mj_forward(m, d);
    printf("nqpos: %i\n", m->nq);
    printf("orig pos: %f\trecon pos: %f\n", d->qpos[1], d->qpos[36]);
    // return 1;
    // init GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }
    
    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // Set initial states
    // mju_copy(&d->qpos[0], orig_states[0].data(), 35);
    // mju_copy(&d->qpos[35], recon_states[0].data(), 35);

    // run main loop, target real-time simulation and 60 fps rendering
    while( !glfwWindowShouldClose(window) )
    {
        // If not paused, then advance data
        if (!ispaused && data_ind < traj_len - 1) {
            data_ind += 1;
            mju_copy(&d->qpos[0], orig_states[data_ind].data(), 35);
            mju_copy(&d->qpos[35], recon_states[data_ind].data(), 35);
            printf("data_ind: %i\n", data_ind);
            mj_forward(m, d);
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
        this_thread::sleep_for(chrono::milliseconds(1/60*100));
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}
