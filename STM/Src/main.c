/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "oled.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "stm32f4xx_hal.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

I2C_HandleTypeDef hi2c2;

TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim6;
TIM_HandleTypeDef htim8;
TIM_HandleTypeDef htim9;
TIM_HandleTypeDef htim12;

UART_HandleTypeDef huart3;

/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* Definitions for moveCarTask */
osThreadId_t moveCarTaskHandle;
const osThreadAttr_t moveCarTask_attributes = {
  .name = "moveCarTask",
  .stack_size = 1024 * 4,
  .priority = (osPriority_t) osPriorityNormal4,
};
/* Definitions for encoderTask */
osThreadId_t encoderTaskHandle;
const osThreadAttr_t encoderTask_attributes = {
  .name = "encoderTask",
  .stack_size = 256 * 4,
  .priority = (osPriority_t) osPriorityNormal2,
};
/* Definitions for oledTask */
osThreadId_t oledTaskHandle;
const osThreadAttr_t oledTask_attributes = {
  .name = "oledTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityLow,
};
/* Definitions for ultraTask */
osThreadId_t ultraTaskHandle;
const osThreadAttr_t ultraTask_attributes = {
  .name = "ultraTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityNormal3,
};
/* Definitions for communicateTask */
osThreadId_t communicateTaskHandle;
const osThreadAttr_t communicateTask_attributes = {
  .name = "communicateTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityLow,
};
/* Definitions for gyroTask */
osThreadId_t gyroTaskHandle;
const osThreadAttr_t gyroTask_attributes = {
  .name = "gyroTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityNormal3,
};
/* Definitions for infraTask */
osThreadId_t infraTaskHandle;
const osThreadAttr_t infraTask_attributes = {
  .name = "infraTask",
  .stack_size = 256 * 4,
  .priority = (osPriority_t) osPriorityNormal3,
};
/* USER CODE BEGIN PV */
//Wheel details
#define LEFT_COUNTS_PER_CM   73.32f
#define RIGHT_COUNTS_PER_CM  74.50f
#define SERVO_PWM_MIN 94u
#define SERVO_PWM_MAX 235u
#define SERVO_CENTER  150u

//Servo motor stuff
uint16_t servoValue = 150;
uint16_t servoRight = 235;
uint16_t servoLeft = 94;

static volatile float turnValue = 0;

static volatile float d = 0.0f;
static volatile float tc_target_yaw = 0.f;

static volatile uint32_t tc_t0_ms = 0;
static volatile uint32_t tc_stable_t0 = 0;
// yaw PID (PD is usually enough for heading)
static const float TC_DEADBAND_DEG = 1.2f;  // inside this, we consider "at target"
static float tc_prev_err = 0.f;
// +1 = left, -1 = right; +1 = forward, -1 = backward
static volatile int tc_dir_lr = +1;
static volatile int tc_dir_fb = +1;
static uint32_t tc_brake_until_ms = 0;

static float tc_kp = 0.040f;      // start small; tune on floor
static float tc_kd = 0.007f;      // derivative on yaw error
static const float DUTY_MIN_MOVE = 0.10f;   // just above friction
static const float DUTY_MAX_CAP  = 0.80f;   // absolute safety cap
static const uint32_t TC_STABLE_MS = 120;   // must remain inside tol this long
static const uint32_t TC_TIMEOUT_MS = 1500; // safety
static const uint32_t TC_TIMEOUT_MS2 = 3000; // safety
static volatile uint32_t mc_t0 = 0;

static volatile int direction = 0;

static bool L_in = false, R_in = false;
static volatile int32_t mc_prev_eL = 0, mc_prev_eR = 0;
static volatile bool directionayman = true;

//Gyro stuff
// Gyro full-scale ranges (GYRO_CONFIG_1 [2:1] bits)
#define GYRO_RATE_250   0x00  // ±250 dps
#define GYRO_RATE_500   0x02  // ±500 dps
#define GYRO_RATE_1000  0x04  // ±1000 dps
#define GYRO_RATE_2000  0x06  // ±2000 dps

// Gyro low-pass filter settings (GYRO_CONFIG_1 [5:3] bits)
#define GYRO_LPF_196HZ  0x00
#define GYRO_LPF_151HZ  0x08
#define GYRO_LPF_119HZ  0x10
#define GYRO_LPF_51HZ   0x18
#define GYRO_LPF_23HZ   0x20
#define GYRO_LPF_11HZ   0x28
#define GYRO_LPF_5HZ    0x30
#define GYRO_LPF_17HZ   0x38   // this is the one your code is calling

static float gz_bias = 0.0f;
const  float GYRO_LSB_PER_DPS = 131.072f; // ±250 dps
static float yaw_deg = 0.0f;
static  float yaw_alpha = 0.99f;      // 0.95..0.995 (higher = smoother, more drift)
static float mag_bx=0, mag_by=0, mag_bz=0;   // hard-iron
static float mag_sx=1, mag_sy=1, mag_sz=1;   // soft-iron
static int yaw_inited = 0;
static float heading_offset_deg = 0.f;

typedef enum { TC_IDLE=0, Phase1, Phase2, MC_RUNNING, TC_TURNING, CORRECTION, CORRECTION2, WF_RUNNING } tc_state_t;
static volatile tc_state_t tc_state = TC_IDLE;

static volatile int32_t mc_tL=0, mc_tR=0;     // targets in counts
static volatile float   mc_dutyL=0.40f;       // default fixed duty
static volatile float   mc_dutyR=0.40f;
static const uint32_t   MC_TIMEOUT_MS = 17000; // safety timeout per move

//Ultrasonic stuff

volatile uint32_t ic_start=0;
volatile uint32_t echo_us=0;
volatile uint8_t ic_captured=0;
volatile uint8_t waiting_for_falling=0;
volatile float distance_cm = 0;


//Ir sensor stuff
volatile float g_ir_left_cm  = 0.0f;       // Left sensor (PA3 / ADC1_IN3)
volatile float g_ir_right_cm = 0.0f;       // Right sensor (PA2 / ADC1_IN2)

// ===================== [ADD] Wall-follow constants & state =====================
typedef enum { WF_LEFT=0, WF_RIGHT=1 } wf_side_t;

static volatile wf_side_t wf_side = WF_LEFT;
static volatile float wf_set_cm    = 20.0f;   // desired distance to the wall
static volatile float wf_base_duty = 0.25f;   // forward base duty (0.12..0.40)


static const float    WF_MIN_CM = 7.3f;       // ignore values below this
static const float    WF_MAX_CM = 55.0f;      // consider "no wall" if above this
static const uint32_t WF_LOST_HOLD_MS = 120;  // must be "lost" this long to stop
static uint32_t       wf_lost_since_ms = 0;   // 0 = currently valid

// Distance measurement while wall-following
static volatile int32_t wf_encA0 = 0;
static volatile int32_t wf_encB0 = 0;
static volatile float   wf_dist_cm = 0.0f;

// --- Lock-to-current (IR) support ---
static volatile bool     wf_lock_pending   = false;  // true until we freeze wf_set_cm
static uint32_t          wf_lock_t0_ms     = 0;      // when locking started
static uint8_t           wf_lock_good_cnt  = 0;      // consecutive valid samples
static const uint32_t    WF_LOCK_ACQUIRE_MS = 300;   // wait this long to stabilize
static float             wf_lock_last_set  = 0.0f;   // last locked setpoint (debug)

// --- Anti-jerk / smoothing ---
static float   wf_prev_cm        = 0.0f;     // filtered distance (cm)
static bool    wf_prev_valid     = false;
static uint16_t wf_servo_prev    = SERVO_CENTER;
static float   wf_cmdL_prev      = 0.0f;
static float   wf_cmdR_prev      = 0.0f;

static volatile float Xf = 0.0f;

// --- WF tuning (globals you can tweak at runtime) ---
static volatile float    wf_base_slow        = 0.20f;   // was 0.18
static volatile float    wf_min_slow         = 0.12f;   // same
static volatile float    wf_slow_window_cm   = 22.0f;   // was 15.9 (less taper = snappier)

static volatile float    wf_near_cm          = 3.2f;    // was 3.0  (only soften when very close)
static volatile float    wf_servo_db_cm      = 1.2f;    // was 1.2  (respond to smaller errors)

static volatile float    wf_kp_servo         = 2.0f;    // was 2.2  (stronger steering)
static volatile float    wf_kp_diff          = 0.10f;   // was 0.10 (more wheel bias)

static volatile float    wf_kp_servo_near    = 0.60f;   // was 0.40
static volatile float    wf_kp_diff_near     = 0.50f;   // was 0.40

static volatile float    wf_diff_clamp       = 0.06f;   // was 0.04 (allow more L/R delta)

static volatile uint16_t wf_servo_slew       = 4;       // was 3  (faster servo ramp)
static volatile uint16_t wf_servo_slew_near  = 2;       // was 1

static volatile float    wf_duty_slew        = 0.045f;  // was 0.02 (faster duty updates)
static volatile float    wf_duty_slew_near   = 0.025f;  // was 0.04

// Anti-jerk / smoothing
static float   wf_alpha          = 0.35f;    // was 0.35 — trust new sample more (less lag)
static float   wf_jump_cm        = 4.0f;     // same


// brief-glitch cruise
static volatile float    wf_glitch_cruise    = 0.12f;   // forward duty while waiting
static volatile float    wf_glitch_slew      = 0.008f;  // slew while waiting

// Remember length C (cm) from the last WL999/WR999 run
static volatile float wf_last_len_cm = NAN;
static volatile bool  wf_curr_run_is_999 = false;

// --- Acceptance band (deadband + hysteresis) ---
static volatile float wf_ok_in_cm   = 2.0f;  // enter "OK" band when |err| <= 2 cm
static volatile float wf_ok_out_cm  = 2.8f;  // leave "OK" band when |err| >= 3 cm
static volatile float wf_ok_min_fw  = 0.16f; // straight cruise speed while inside band
static bool wf_ok_active = false;            // state: currently inside the band



// helper
static inline float round_cm(float x) { return floorf(x + 0.5f); }

// round float cm to nearest integer cm (handles negatives correctly)
static inline int round_cm_i(float x) { return (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f); }


static inline float slew_f(float prev, float target, float step){
    float d = target - prev;
    if (d > step)  d = step;
    if (d < -step) d = -step;
    return prev + d;
}
static inline uint16_t slew_u16(uint16_t prev, uint16_t target, uint16_t step){
    if (target > prev){
        uint16_t d = target - prev;
        if (d > step) d = step;
        return prev + d;
    } else {
        uint16_t d = prev - target;
        if (d > step) d = step;
        return prev - d;
    }
}


static inline float clampf(float v, float lo, float hi){
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

//Encoder stuff
static uint16_t encoder_lastA = 0;
static uint16_t encoder_lastB = 0;
//Encoder stuff

// Difference
int32_t encoder_deltaA = 0;
int32_t encoder_deltaB = 0;
// Counts per second
volatile float encoder_cpsA = 0;
volatile float encoder_cpsB = 0;

volatile int32_t encA_pos = 0;  // left wheel
volatile int32_t encB_pos = 0;  // right wheel


volatile int32_t object1dleft = 0;
volatile int32_t object1dright = 0;
volatile int32_t object2dleft = 0;
volatile int32_t object2dright = 0;

volatile float differenceobject1 = 0.0f;
volatile float differenceobject2 = 0.0f;

static volatile int32_t numObstacle = 1;

static float straightAngle = 0.0f;
static const float straightError = 0.6f;
//Communicate stuff
//Uart stuff
uint8_t aRxBuffer[5] = {0}; //Fixed buffer of 5 bytes
static volatile int rxReady = 0; //Flag

static inline void set_pwm(TIM_HandleTypeDef* h, uint32_t ch, uint16_t val);
static inline void hardBrake(void);


//Homee compute
static inline float compute_homee_Y(void);
static void RunHOMEE(void);

// --- Debug/publication of computed distances ---
static volatile float g_homee_last_f = NAN;
static volatile int   g_homee_last_i = 0;
static volatile float g_dumee_last_f = NAN;
static volatile int   g_dumee_last_i = 0;


static inline int is_digit(uint8_t c) { return (c >= '0' && c <= '9'); }
static int parse_magnitude(const uint8_t *b) {
  // b[0..2] are ASCII digits D2 D1 D0 => 100*D2 + 10*D1 + D0
  if (!is_digit(b[0]) || !is_digit(b[1]) || !is_digit(b[2])) return 0;
  return (b[0]-'0')*100 + (b[1]-'0')*10 + (b[2]-'0');
}

void MoveRightWheel(float right_cmd){

	if (!isnan(right_cmd)) {
		if (right_cmd >  1.f) right_cmd = 1.f;
		if (right_cmd < -1.f) right_cmd = -1.f;
	}


	const uint16_t arr9 = (uint16_t)htim9.Init.Period;
	//Fabsf always return a positive (absolute) value for the "speed" of the car rotation
	uint16_t dutyRight = (uint16_t)(fabsf(right_cmd) * arr9);
	//Move Forward

	    if (!isnan(right_cmd)) {
	        uint16_t dutyRight = (uint16_t)(fabsf(right_cmd) * arr9);
	        if (right_cmd >= 0.f) {
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_1, 0);
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_2, dutyRight);
	        	set_pwm(&htim4, TIM_CHANNEL_3, (uint16_t)htim4.Init.Period);
	        	set_pwm(&htim4, TIM_CHANNEL_4,(uint16_t)htim4.Init.Period );
	        } else {
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_2, 0);
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_1, dutyRight);
	        	set_pwm(&htim4, TIM_CHANNEL_3, (uint16_t)htim4.Init.Period);
	        	set_pwm(&htim4, TIM_CHANNEL_4,(uint16_t)htim4.Init.Period );
	        }

	    }
}


// left/right_cmd is a percentage, for example 0.9 means 90% speed forward
void MoveLeftWheel(float left_cmd){

	//Safety Guards to prevent invalid values
	if (!isnan(left_cmd)) {
	        if (left_cmd >  1.f) left_cmd = 1.f;
	        if (left_cmd < -1.f) left_cmd = -1.f;
	    }

	//Get the constants to calculate duty
	const uint16_t arr4 = (uint16_t)htim4.Init.Period;
	//Fabsf always return a positive (absolute) value for the "speed" of the car rotation
	uint16_t dutyLeft = (uint16_t)(fabsf(left_cmd) * arr4);
	//Move Forward

	 if (!isnan(left_cmd)) {
	        uint16_t dutyLeft = (uint16_t)(fabsf(left_cmd) * arr4);
	        if (left_cmd >= 0.f) {
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_3, 0);
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_4, dutyLeft);
	        	set_pwm(&htim9, TIM_CHANNEL_2, (uint16_t)htim9.Init.Period);
	        	set_pwm(&htim9, TIM_CHANNEL_1, (uint16_t)htim9.Init.Period);
	        } else {
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_4, 0);
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_3, dutyLeft);
	        	set_pwm(&htim9, TIM_CHANNEL_2, (uint16_t)htim9.Init.Period);
	        	set_pwm(&htim9, TIM_CHANNEL_1, (uint16_t)htim9.Init.Period);
	        }
	    }

}

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C2_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM6_Init(void);
static void MX_TIM8_Init(void);
static void MX_TIM9_Init(void);
static void MX_TIM12_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM4_Init(void);
void StartDefaultTask(void *argument);
void MoveCarTask(void *argument);
void EncoderTask(void *argument);
void OledTask(void *argument);
void UltraTask(void *argument);
void CommunicateTask(void *argument);
void GyroTask(void *argument);
void InfraTask(void *argument);

/* USER CODE BEGIN PFP */


/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

//Functions for angles
static inline float wrap360(float a){
  if (a >= 360.0f) a -= 360.0f;
  else if (a < 0.0f) a += 360.0f;
  return a;
}
static inline float get_heading_deg(void) {          // virtual heading for UI/commands
    return wrap360((float)(yaw_deg + heading_offset_deg));
}
static inline void set_heading_baseline(float desired_deg) { // make current yaw read desired
    heading_offset_deg = wrap360(desired_deg - yaw_deg);
}
static inline float unwrap_towards(float target, float ref){
  // Shift 'target' by ±360 so it's closest to 'ref'
  while (target - ref >  180.0f) target -= 360.0f;
  while (target - ref < -180.0f) target += 360.0f;
  return target;
}
static inline float countsToCmL(int32_t counts){ return counts / LEFT_COUNTS_PER_CM; }
static inline float countsToCmR(int32_t counts){ return counts / RIGHT_COUNTS_PER_CM; }
static inline int32_t cmToCountsL(float cm){ return (int32_t)(cm * LEFT_COUNTS_PER_CM + 0.5f); }
static inline int32_t cmToCountsR(float cm){ return (int32_t)(cm * RIGHT_COUNTS_PER_CM + 0.5f); }

//Communication
//Communication
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart){
	if (huart->Instance == USART3) {
	    rxReady = 1; // tell a task new data arrived
	    HAL_UART_Receive_IT(&huart3, aRxBuffer, 5);// immediately re-arm for the next 5-byte frame
	  }
}
void snap(void){
    if(numObstacle == 1){
    	int32_t leftDistance = encA_pos - object1dleft;
    	int32_t rightDistance = encB_pos - object1dright;
    	differenceobject1 = (countsToCmL(leftDistance) + countsToCmR(rightDistance))/2;
    }
    else{
    	int32_t leftDistance = encA_pos - object2dleft;
    	int32_t rightDistance = encB_pos - object2dright;
    	differenceobject2 = (countsToCmL(leftDistance) + countsToCmR(rightDistance))/2;
    }
    numObstacle++;
}

void SampleEncoder(float dt_sec){
	uint32_t nowA = (uint16_t)__HAL_TIM_GET_COUNTER(&htim2);
	encoder_deltaA = (int16_t)(nowA - encoder_lastA);
	encoder_lastA = nowA;
	encA_pos += (int32_t)encoder_deltaA;
	// Get the difference, divide by the time (Actually not used by the video)
	if (dt_sec > 0.0f) encoder_cpsA = encoder_deltaA / dt_sec;

	uint32_t nowB = (uint16_t)__HAL_TIM_GET_COUNTER(&htim3);
	int16_t deltaB_raw = (int16_t)(nowB - encoder_lastB);
	encoder_lastB = nowB;

	encoder_deltaB = (int16_t)(-deltaB_raw);
	encB_pos += (int32_t)encoder_deltaB;
	if (dt_sec > 0.0f) encoder_cpsB = (float)encoder_deltaB / dt_sec;

//	char buffer[30];
//	snprintf(buffer, sizeof buffer, "%.2f,%.2f\r\n", (double)encA_pos, (double)encB_pos);
//	HAL_UART_Transmit(&huart3, buffer, strlen(buffer), HAL_MAX_DELAY);


}
//Functions for movement
//Move Left Wheel Forward
void motorA_forward(uint16_t duty){
	set_pwm(&htim4, TIM_CHANNEL_3, 0);
	set_pwm(&htim4, TIM_CHANNEL_4, duty);
}
//Move Left Wheel Backward
void motorA_backward(uint16_t duty){
	set_pwm(&htim4, TIM_CHANNEL_4, 0);
	set_pwm(&htim4, TIM_CHANNEL_3, duty);
}
//Move Right Wheel Backward
void motorB_backward(uint16_t duty){
	set_pwm(&htim9, TIM_CHANNEL_2, 0);
	set_pwm(&htim9, TIM_CHANNEL_1, duty);
}
//Move Right Wheel Forward
void motorB_forward(uint16_t duty){
	set_pwm(&htim9, TIM_CHANNEL_1, 0);
	set_pwm(&htim9, TIM_CHANNEL_2, duty);
}
void MoveCar(float left_cmd, float right_cmd){

	//Safety Guards to prevent invalid values
	if (!isnan(left_cmd)) {
	        if (left_cmd >  1.f) left_cmd = 1.f;
	        if (left_cmd < -1.f) left_cmd = -1.f;
	    }
	if (!isnan(right_cmd)) {
		if (right_cmd >  1.f) right_cmd = 1.f;
		if (right_cmd < -1.f) right_cmd = -1.f;
	}

	//Get the constants to calculate duty
	const uint16_t arr4 = (uint16_t)htim4.Init.Period;
	const uint16_t arr9 = (uint16_t)htim9.Init.Period;
	//Fabsf always return a positive (absolute) value for the "speed" of the car rotation
	uint16_t dutyLeft = (uint16_t)(fabsf(left_cmd) * arr4);
	uint16_t dutyRight = (uint16_t)(fabsf(right_cmd) * arr9);
	//Move Forward

	 if (!isnan(left_cmd)) {
	        uint16_t dutyLeft = (uint16_t)(fabsf(left_cmd) * arr4);
	        if (left_cmd >= 0.f) {
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_3, 0);
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_4, dutyLeft);
	        } else {
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_4, 0);
	            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_3, dutyLeft);
	        }
	    }

	    if (!isnan(right_cmd)) {
	        uint16_t dutyRight = (uint16_t)(fabsf(right_cmd) * arr9);
	        if (right_cmd >= 0.f) {
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_1, 0);
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_2, dutyRight);
	        } else {
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_2, 0);
	            __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_1, dutyRight);
	        }

	    }
}

static inline uint16_t servo_from_err_fb(float err_deg, int fb_dir){
    // fb_dir: +1 forward, -1 backward
    // When backing up, flip sign so the car backs into the same yaw change.
    const float K_SERVO = 1.5f;             // tune 0.6..1.5
    float eff_err = (fb_dir >= 0) ? err_deg : -err_deg;
    float pwm = SERVO_CENTER + K_SERVO * eff_err;
    if (pwm < SERVO_PWM_MIN) pwm = SERVO_PWM_MIN;
    if (pwm > SERVO_PWM_MAX) pwm = SERVO_PWM_MAX;
    return (uint16_t)pwm;
}

static inline uint16_t servo_from_err_fbright(float err_deg, int fb_dir){
    // fb_dir: +1 forward, -1 backward
    // When backing up, flip sign so the car backs into the same yaw change.
    const float K_SERVO = 0.5f;             // tune 0.6..1.5 //We liked 0.4
    float eff_err = (fb_dir >= 0) ? err_deg : -err_deg;
    float pwm = SERVO_CENTER + K_SERVO * eff_err;
    if (pwm < SERVO_PWM_MIN) pwm = SERVO_PWM_MIN;
    if (pwm > SERVO_PWM_MAX) pwm = SERVO_PWM_MAX;
    return (uint16_t)pwm;
}
static inline uint16_t servo_from_err_fbleft(float err_deg, int fb_dir){
    // fb_dir: +1 forward, -1 backward
    // When backing up, flip sign so the car backs into the same yaw change.
    const float K_SERVO = 0.4f;             // tune 0.6..1.5 //We liked 0.4
    float eff_err = (fb_dir >= 0) ? err_deg : -err_deg;
    float pwm = SERVO_CENTER + K_SERVO * eff_err;
    if (pwm < SERVO_PWM_MIN) pwm = SERVO_PWM_MIN;
    if (pwm > SERVO_PWM_MAX) pwm = SERVO_PWM_MAX;
    return (uint16_t)pwm;
}


static inline uint16_t servo_from_err_fb3(float err_deg, int fb_dir){
    // fb_dir: +1 forward, -1 backward
    // When backing up, flip sign so the car backs into the same yaw change.
    const float K_SERVO = 1.5f;             // tune 0.6..1.5
    float eff_err = (fb_dir >= 0) ? err_deg : -err_deg;
    float pwm = SERVO_CENTER + K_SERVO * eff_err;
    if (pwm < SERVO_PWM_MIN) pwm = SERVO_PWM_MIN;
    if (pwm > SERVO_PWM_MAX) pwm = SERVO_PWM_MAX;
    return (uint16_t)pwm;
}


static inline uint16_t servo_from_err_fb5(float err_deg, int fb_dir){
    // fb_dir: +1 forward, -1 backward
    // When backing up, flip sign so the car backs into the same yaw change.
    const float K_SERVO = 2.0f;             // tune 0.6..1.5
    float eff_err = (fb_dir >= 0) ? err_deg : -err_deg;
    float pwm = SERVO_CENTER + K_SERVO * eff_err;
    if (pwm < SERVO_PWM_MIN) pwm = SERVO_PWM_MIN;
    if (pwm > SERVO_PWM_MAX) pwm = SERVO_PWM_MAX;
    return (uint16_t)pwm;
}



//Functions for movements
//Function to compare and
static inline void set_pwm(TIM_HandleTypeDef* h, uint32_t ch, uint16_t val){
	//Compare with CCR value to generate PWM Signal that goes to the motor driver
	__HAL_TIM_SET_COMPARE(h, ch, val);
}
static inline void hardBrake(void){
	set_pwm(&htim4, TIM_CHANNEL_3, (uint16_t)htim4.Init.Period);
	set_pwm(&htim4, TIM_CHANNEL_4,(uint16_t)htim4.Init.Period );
	set_pwm(&htim9, TIM_CHANNEL_2, (uint16_t)htim9.Init.Period);
	set_pwm(&htim9, TIM_CHANNEL_1, (uint16_t)htim9.Init.Period);
}

void start_arc_turn(char lr, char fb, float deg)
{

	if (lr == 'L'){
		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoLeft);
	}
	else{
		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoRight);
	}
	// Current "virtual" heading (raw + offset)
    float curr_h = get_heading_deg();

    // Target in virtual frame: L increases, R decreases
    float tgt_h = (fb=='F')
                  ? ((lr=='L') ? (curr_h - fabsf(deg)) : (curr_h + fabsf(deg)))
                  : ((lr=='L') ? (curr_h + fabsf(deg)) : (curr_h - fabsf(deg)));
    tgt_h = wrap360(tgt_h);

    // Convert virtual target → raw yaw for the PD loop:
    // virtual = raw + offset  => raw = virtual - offset
    float tgt_raw = wrap360(tgt_h - heading_offset_deg);
    tgt_raw = unwrap_towards(tgt_raw, yaw_deg);   // choose nearest equivalent


    taskENTER_CRITICAL();
    tc_target_yaw = tgt_raw;
    tc_prev_err   = 0.f;
    tc_dir_lr     = (lr=='L') ? +1 : -1;
    tc_dir_fb     = (fb=='F') ? +1 : -1;
    tc_t0_ms      = HAL_GetTick();
    tc_stable_t0  = 0;
    tc_state      = TC_TURNING;
    taskEXIT_CRITICAL();

}

void start_move_straight(float distance_cm, float requested_vmax)
{
    __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);



    taskENTER_CRITICAL();

    mc_tL = encA_pos + cmToCountsL(distance_cm);
    mc_tR = encB_pos + cmToCountsR(distance_cm);

    straightAngle = yaw_deg;

    mc_t0 = HAL_GetTick();
    // NEW: disarm crossing detector + any leftover brake windows
    mc_prev_eL = mc_tL - encA_pos;
    mc_prev_eR = mc_tR - encB_pos;
    tc_state = MC_RUNNING;
    taskEXIT_CRITICAL();
}



// ===================== [ADD] Start wall-follow helper =====================
static void start_wall_follow(wf_side_t side, float set_cm, float base_duty)
{
    __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue); // center steering

    taskENTER_CRITICAL();
    wf_side       = side;
    wf_base_duty  = clampf(base_duty, 0.12f, 0.40f);

    wf_encA0 = encA_pos;
    wf_encB0 = encB_pos;
    wf_dist_cm = 0.0f;

    wf_lost_since_ms = 0;

    wf_prev_valid  = false;
    wf_prev_cm     = 0.0f;
    wf_servo_prev  = SERVO_CENTER;
    wf_cmdL_prev   = 0.0f;
    wf_cmdR_prev   = 0.0f;

    wf_ok_active = false;   // [ADD] start "not in band"

    // (set_cm < 0) means "lock to current" => invoked by WL999 / WR999
    wf_curr_run_is_999 = (set_cm < 0.0f);

    if (set_cm < 0.0f) {
        wf_lock_pending  = true;
        wf_lock_t0_ms    = HAL_GetTick();
        wf_lock_good_cnt = 0;
        wf_set_cm        = 20.0f;  // placeholder until lock
    } else {
        wf_lock_pending  = false;
        wf_set_cm        = round_cm(set_cm);
    }

    tc_state = WF_RUNNING;
    taskEXIT_CRITICAL();
}


static inline float compute_homee_Y(void) {
    // 23 + 10 + 19.5 + 23 - 30 = 45.5
    return differenceobject1 + differenceobject2 + 23.0f + 10.0f + 19.5f + 23.0f - 25; // cm
}

static void RunHOMEE(void) {
    float Yf = compute_homee_Y();          // may be fractional (e.g., 45.5)
    if (!isfinite(Yf)) return;



//    // publish + print
//    g_homee_last_f = Yf;
//    {
//        char msg[48];
//        int y10 = (int)lrintf(Yf * 10.f);
//        int n = snprintf(msg, sizeof msg, "HOME: %d.%d\r\n", y10/10, abs(y10%10));
//        HAL_UART_Transmit(&huart3, (uint8_t*)msg, n, HAL_MAX_DELAY);
//    }

    directionayman = (Yf >= 0);            // forward if >= 0, else reverse
    start_move_straight((float)Yf, 0.0f);  // API is float, but value is an int
}

static inline float compute_dumee_X(void) {
    float C = wf_last_len_cm;             // last WL999/WR999 length in cm
    //if (!isfinite(C)) return -10.0f;        // no prior 999-run -> safe no-op distance
//    return fmaxf(5, ((1.2f * C)/2 - 10.0f));              // X = C/2 - 45
    return (fmaxf(15, C/2) + 35 -50 + 10);

}


static void RunDUMEE(void) {
    Xf = compute_dumee_X();          // may be fractional

    if (!isfinite(Xf)) return;

//    int cm = round_cm_i(Xf);

//    // publish + print
//    g_dumee_last_f = Xf;
//    g_dumee_last_i = cm;
//    {
//        char msg[48];
//        int x10 = (int)lrintf(Xf * 10.f);
//        int n = snprintf(msg, sizeof msg, "DUMEE: %d.%d -> %dcm\r\n",
//                         x10/10, abs(x10%10), cm);
//        HAL_UART_Transmit(&huart3, (uint8_t*)msg, n, HAL_MAX_DELAY);
//    }

    directionayman = (Xf >= 0);
    start_move_straight((float)Xf, 0.0f);
}





void SF(float distance_cm)
{
    if (!isfinite(distance_cm)) return;

    // Round to nearest integer cm (19.5 -> 20, -19.5 -> -20)
    int cm = (distance_cm >= 0.0f)
           ? (int)(distance_cm + 0.5f)
           : (int)(distance_cm - 0.5f);

    directionayman = (cm >= 0);

    // API still takes float, but it's an integer value
    start_move_straight((float)cm, 0.0f);
}

//Function for ultrasonic


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C2_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM6_Init();
  MX_TIM8_Init();
  MX_TIM9_Init();
  MX_TIM12_Init();
  MX_USART3_UART_Init();
  MX_ADC1_Init();
  MX_TIM4_Init();
  /* USER CODE BEGIN 2 */
  ICM_LockInit();
  //Motor PWM Start
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_4);
  HAL_TIM_PWM_Start(&htim9, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim9, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim12, TIM_CHANNEL_2);

  //Encoder stuff
  //Encoder Start
  HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL);
  HAL_TIM_Encoder_Start(&htim3, TIM_CHANNEL_ALL);
  // Get the initial last value
  encoder_lastA = __HAL_TIM_GET_COUNTER(&htim2);
  encoder_lastB = __HAL_TIM_GET_COUNTER(&htim3);

  // ultrasonic
  HAL_TIM_Base_Start(&htim6);                      // 1 µs tick timebase
  HAL_TIM_IC_Start_IT(&htim8, TIM_CHANNEL_3);      // ECHO on TIM8_CH3
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET); // TRIG low

  //Communication
  HAL_UART_Receive_IT(&huart3,(uint8_t *) aRxBuffer,5);//Collect 5 bytes into the buffer and then call back the interrupt when ready

  /* USER CODE END 2 */

  /* Init scheduler */
  osKernelInitialize();

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* creation of moveCarTask */
  moveCarTaskHandle = osThreadNew(MoveCarTask, NULL, &moveCarTask_attributes);

  /* creation of encoderTask */
  encoderTaskHandle = osThreadNew(EncoderTask, NULL, &encoderTask_attributes);

  /* creation of oledTask */
  oledTaskHandle = osThreadNew(OledTask, NULL, &oledTask_attributes);

  /* creation of ultraTask */
  ultraTaskHandle = osThreadNew(UltraTask, NULL, &ultraTask_attributes);

  /* creation of communicateTask */
  communicateTaskHandle = osThreadNew(CommunicateTask, NULL, &communicateTask_attributes);

  /* creation of gyroTask */
  gyroTaskHandle = osThreadNew(GyroTask, NULL, &gyroTask_attributes);

  /* creation of infraTask */
  infraTaskHandle = osThreadNew(InfraTask, NULL, &infraTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

  /* Start scheduler */
  osKernelStart();

  /* We should never get here as control is now taken by the scheduler */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV2;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ENABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 2;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_2;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_480CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_3;
  sConfig.Rank = 2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief I2C2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C2_Init(void)
{

  /* USER CODE BEGIN I2C2_Init 0 */

  /* USER CODE END I2C2_Init 0 */

  /* USER CODE BEGIN I2C2_Init 1 */

  /* USER CODE END I2C2_Init 1 */
  hi2c2.Instance = I2C2;
  hi2c2.Init.ClockSpeed = 100000;
  hi2c2.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c2.Init.OwnAddress1 = 0;
  hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c2.Init.OwnAddress2 = 0;
  hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C2_Init 2 */

  /* USER CODE END I2C2_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 65535;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 10;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 10;
  if (HAL_TIM_Encoder_Init(&htim2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 65535;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 10;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 10;
  if (HAL_TIM_Encoder_Init(&htim3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 15;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 65535;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim4, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */
  HAL_TIM_MspPostInit(&htim4);

}

/**
  * @brief TIM6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM6_Init(void)
{

  /* USER CODE BEGIN TIM6_Init 0 */

  /* USER CODE END TIM6_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM6_Init 1 */

  /* USER CODE END TIM6_Init 1 */
  htim6.Instance = TIM6;
  htim6.Init.Prescaler = 16-1;
  htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim6.Init.Period = 65535;
  htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim6) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim6, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM6_Init 2 */

  /* USER CODE END TIM6_Init 2 */

}

/**
  * @brief TIM8 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM8_Init(void)
{

  /* USER CODE BEGIN TIM8_Init 0 */

  /* USER CODE END TIM8_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_IC_InitTypeDef sConfigIC = {0};

  /* USER CODE BEGIN TIM8_Init 1 */

  /* USER CODE END TIM8_Init 1 */
  htim8.Instance = TIM8;
  htim8.Init.Prescaler = 16-1;
  htim8.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim8.Init.Period = 65535;
  htim8.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim8.Init.RepetitionCounter = 0;
  htim8.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim8) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim8, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_IC_Init(&htim8) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim8, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigIC.ICPolarity = TIM_INPUTCHANNELPOLARITY_BOTHEDGE;
  sConfigIC.ICSelection = TIM_ICSELECTION_DIRECTTI;
  sConfigIC.ICPrescaler = TIM_ICPSC_DIV1;
  sConfigIC.ICFilter = 0;
  if (HAL_TIM_IC_ConfigChannel(&htim8, &sConfigIC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM8_Init 2 */

  /* USER CODE END TIM8_Init 2 */

}

/**
  * @brief TIM9 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM9_Init(void)
{

  /* USER CODE BEGIN TIM9_Init 0 */

  /* USER CODE END TIM9_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM9_Init 1 */

  /* USER CODE END TIM9_Init 1 */
  htim9.Instance = TIM9;
  htim9.Init.Prescaler = 15;
  htim9.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim9.Init.Period = 65535;
  htim9.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim9.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim9) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim9, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim9) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim9, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim9, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM9_Init 2 */

  /* USER CODE END TIM9_Init 2 */
  HAL_TIM_MspPostInit(&htim9);

}

/**
  * @brief TIM12 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM12_Init(void)
{

  /* USER CODE BEGIN TIM12_Init 0 */

  /* USER CODE END TIM12_Init 0 */

  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM12_Init 1 */

  /* USER CODE END TIM12_Init 1 */
  htim12.Instance = TIM12;
  htim12.Init.Prescaler = 160;
  htim12.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim12.Init.Period = 1000;
  htim12.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim12.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_PWM_Init(&htim12) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim12, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM12_Init 2 */

  /* USER CODE END TIM12_Init 2 */
  HAL_TIM_MspPostInit(&htim12);

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED3_GPIO_Port, LED3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, IMU_Select_Pin|US_TRIG_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, OLED_DC_Pin|OLED_RES_Pin|OLED_SDA_Pin|OLED_SCL_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : LED3_Pin */
  GPIO_InitStruct.Pin = LED3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED3_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : IMU_Select_Pin US_TRIG_Pin */
  GPIO_InitStruct.Pin = IMU_Select_Pin|US_TRIG_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : OLED_DC_Pin OLED_RES_Pin OLED_SDA_Pin OLED_SCL_Pin */
  GPIO_InitStruct.Pin = OLED_DC_Pin|OLED_RES_Pin|OLED_SDA_Pin|OLED_SCL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : button_Pin */
  GPIO_InitStruct.Pin = button_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(button_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
static inline void us_delay(uint32_t us){
  __HAL_TIM_SET_COUNTER(&htim6, 0);
  while (__HAL_TIM_GET_COUNTER(&htim6) < us) {}
}


void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance == TIM8 && htim->Channel == HAL_TIM_ACTIVE_CHANNEL_3)
  {
    uint32_t now = HAL_TIM_ReadCapturedValue(&htim8, TIM_CHANNEL_3);

    if (!waiting_for_falling) {
      ic_start = now;
      waiting_for_falling = 1;
      __HAL_TIM_SET_CAPTUREPOLARITY(&htim8, TIM_CHANNEL_3, TIM_INPUTCHANNELPOLARITY_FALLING);
    } else {
      uint32_t end = now;
      uint32_t delta = (end >= ic_start) ? (end - ic_start) : (0x10000u - ic_start + end);
      echo_us = delta;         // 1 tick = 1 µs @ PSC=16-1, HSI16
      ic_captured = 1;
      waiting_for_falling = 0;
      __HAL_TIM_SET_CAPTUREPOLARITY(&htim8, TIM_CHANNEL_3, TIM_INPUTCHANNELPOLARITY_RISING);
    }
  }
}


/* USER CODE END 4 */

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the defaultTask thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN 5 */
  /* Infinite loop */
	  for(;;)
	  {
		  HAL_GPIO_TogglePin(LED3_GPIO_Port, LED3_Pin);
		  osDelay(500);
	  }
  /* USER CODE END 5 */
}

/* USER CODE BEGIN Header_MoveCarTask */
/**
* @brief Function implementing the moveCarTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_MoveCarTask */
void MoveCarTask(void *argument)
{
  /* USER CODE BEGIN MoveCarTask */
	TickType_t next = xTaskGetTickCount();
	const TickType_t period = pdMS_TO_TICKS(1); // 1 ms loop
  /* Infinite loop */
  for(;;)
  {
	  if(tc_state == Phase1){
		   taskENTER_CRITICAL();      // optional, but guarantees a consistent read
		   d = distance_cm;
		   taskEXIT_CRITICAL();
		  if(d < 30){
			  hardBrake();
			  tc_state = TC_IDLE;
			  const char ack[] = "AAAAA";
			  HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);

		  }
		  else{
			  //CmdL, CmdR
			  float target = unwrap_towards(straightAngle, yaw_deg);
			  float err    = yaw_deg - target;
			  int fb = directionayman ? +1 : -1;
			  bool in_deadband = (fabsf(err) <= straightError);
			  if (in_deadband) {
				  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);   // center
			  } else {
				  uint16_t servo_pwm = servo_from_err_fb3(err, fb);
				  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
			  }
			  MoveCar(0.23, 0.23);
		  }
	      vTaskDelayUntil(&next, period);
	      continue;
	  }
	  else if (tc_state == CORRECTION){
		  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);
		  float temp = 23.0f - distance_cm;
//			  char buf[20];
//			  int n = snprintf(buf, sizeof(buf), "%.2f", (float)-1.0f * (float)temp);
//			  HAL_UART_Transmit(&huart3, (uint8_t*)buf, n, HAL_MAX_DELAY);

			  if (temp < 0){
				  directionayman = true;
			  }
			  else{
				  directionayman = false;
			  }
			  start_move_straight((float)-1.0f * (float)temp, 0.0);


	  }
	  else if (tc_state == CORRECTION2){
		  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);
		  float temp = 10.0f - distance_cm;
//			  char buf[20];
//			  int n = snprintf(buf, sizeof(buf), "%.2f", (float)-1.0f * (float)temp);
//			  HAL_UART_Transmit(&huart3, (uint8_t*)buf, n, HAL_MAX_DELAY);

			  if (temp < 0){
				  directionayman = true;
			  }
			  else{
				  directionayman = false;
			  }
			  start_move_straight((float)-1.0f * (float)temp, 0.0);


	  }
	  else if (tc_state == TC_TURNING) {
	  	      // --- heading error (closest path) ---
	  	      float err_raw_target = unwrap_towards(tc_target_yaw, yaw_deg);
	  	      float err  = yaw_deg - err_raw_target;

	  	      // PD terms (10 ms loop)
	  	      const float dt = 0.001f;
	  	      float derr = (err - tc_prev_err) / dt;   // NOTE: this is essentially yaw rate (dψ/dt)
	  	      tc_prev_err = err;

	  	      uint32_t now = HAL_GetTick();
	  	      bool in_deadband = (fabsf(err) <= TC_DEADBAND_DEG);

//	  	      char msg[100];
//	  	      float abs_err = fabsf(err);

	  	      //sprintf(msg, "TC_DEADBAND_DEG: %.3f, fabsf(err): %.3f\r\n", TC_DEADBAND_DEG, abs_err);

	  	      //HAL_UART_Transmit(&huart3, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);

	  	      // ===================== deadband entry: servo center + HARD BRAKE =====================
	  	      if (in_deadband && tc_stable_t0 == 0) {
	  	          tc_stable_t0      = now;
	  	          tc_brake_until_ms = now + 1000;                     // try 80..150 ms on your floor
	  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);
	  	          hardBrake();
//	  	          char buf[20];
//	  	          int n = snprintf(buf, sizeof(buf), "hardbraked!!!");
//	  	          HAL_UART_Transmit(&huart3, (uint8_t*)buf, n, HAL_MAX_DELAY);
	  	          tc_state = TC_IDLE;
	  	          const char ack[] = "AAAAA";
	  	          HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
	  	          // instantly short both motors
	  	          vTaskDelayUntil(&next, period);
	  	          continue;                                          // let the brake bite this frame
	  	      }

	  	      // ===================== rate-aware brake hold =====================
	  	      // Hold the brake while:
	  	      //  - timed brake window is still active, OR
	  	      //  - we are in the deadband but angular rate is still high
	  	      // This prevents coasting past the target due to inertia.
//	  	      const float RATE_HOLD_DPS = 25.0f;                     // tune 15..40 dps
//	  	      if ((tc_brake_until_ms && now < tc_brake_until_ms) ||
//	  	          (in_deadband && fabsf(derr) > RATE_HOLD_DPS)) {
//	  	    	const char ack[] = "FOUR";
//	  	    	HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
//	  	          // Optionally auto-extend the brake a bit if rate remains high (capped)
//	  	          if (in_deadband && fabsf(derr) > RATE_HOLD_DPS) {
//	  	              uint32_t cap = tc_stable_t0 + 350;             // don’t brake forever; cap extra hold
//	  	              uint32_t ext = now + 60;                       // extend in small chunks
//	  	              if (ext > tc_brake_until_ms) tc_brake_until_ms = (ext < cap) ? ext : tc_brake_until_ms;
//	  	          }
//
//	  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);
//	  	          hardBrake();
//	  	          vTaskDelayUntil(&next, period);
//	  	          continue;
//	  	      } else if (tc_brake_until_ms && now >= tc_brake_until_ms) {
//	  	          tc_brake_until_ms = 0;                             // end of timed brake window
//	  	          hardBrake();                               // release to zero drive
//	  	        const char ack[] = "FIVE";
//	  	        HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
//
//	  	      }

	  	      // ===================== ARC MODE (only drives when NOT in deadband) =====================
	  	      // Pre-slowdown as you approach target (prevents entering band with too much momentum)
	  	      const float BASE_MAG   = 0.32f;                        // nominal base speed
	  	      const float MIN_BASE   = 0.095f;                        // minimum useful base while slowing
	  	      const float SLOW_DEG   = 20.0f;                        // start tapering inside this angle
	  	      const float DIFF_GAIN  = 0.40f;                        // steer-to-differential gain

	  	      float aerr = fabsf(err);
	  	      float base = BASE_MAG;
	  	      if (aerr < SLOW_DEG) {
	  	          // Linear taper: from BASE_MAG at |err|=SLOW_DEG down to MIN_BASE near the band
	  	          float s = aerr / SLOW_DEG;
	  	          base = fmaxf(MIN_BASE, BASE_MAG * s);
	  	      }
	  	      if (in_deadband) hardBrake();                          // absolutely no creep inside band

	  	      float FWD_BASE = (tc_dir_fb > 0) ? +base : -base;

//	  	      // 1) Steering servo: only adjust while actually turning
	  	      if (!in_deadband) {
	  	    	  if( turnValue > 150){
	  	    		  if(direction == 2){
	  	    			uint16_t servo_pwm = servo_from_err_fbleft(err, tc_dir_fb);
	  	    			__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
	  	    		  }
	  	    		  else{
	  	    			uint16_t servo_pwm = servo_from_err_fbright(err, tc_dir_fb);
	  	    			__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
	  	    		  }

	  	    	  }
	  	    	  else if( turnValue == 90){
	  	    		uint16_t servo_pwm = servo_from_err_fb5(err, tc_dir_fb);
	  	    		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
	  	    	  }
	  	    	  else if ( turnValue == 89){
	  	    		uint16_t servo_pwm = servo_from_err_fb5(err, tc_dir_fb);
	  	    		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
	  	    	  }
	  	    	  else{
		  	          uint16_t servo_pwm = servo_from_err_fb(err, tc_dir_fb);
		  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
	  	    	  }
	  	      } else {
	  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, SERVO_CENTER);
	  	      }
	  	      //err > 0 turn right
//	  	      if(!in_deadband){
//	  	    	if( parse_magnitude(&aRxBuffer[2]) > 150 && err > 0){
//	  	    		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoRight);
//	  	    	}
//	  	    	else if (parse_magnitude(&aRxBuffer[2]) > 150 && err > 0){
//	  	    		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoLeft);
//	  	    	}
//	  	    	else if(err > 0){
//	  	    		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoRight);
//	  	    	}
//	  	    	else if(err < 0){
//	  	    		__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoLeft);
//	  	    	}
//
//	  	      }
//	  	      else{
//	  	    	__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, SERVO_CENTER);
//	  	      }

	  	      // 2) Wheel commands: common base ± bias from PD
	  	      float steer_cmd = tc_kp * err + tc_kd * derr;
	  	      float bias = DIFF_GAIN * steer_cmd;
	  	      if (tc_dir_fb < 0) bias = -bias;                       // reversing swaps inside/outside

	  	      // Don’t let bias flip a wheel sign; also forbid creep if base==0
	  	      float max_bias = fabsf(FWD_BASE) - DUTY_MIN_MOVE;
	  	      if (max_bias < 0.0f) max_bias = 0.0f;
	  	      if (bias >  max_bias) bias =  max_bias;
	  	      if (bias < -max_bias) bias = -max_bias;

	  	      float cmdL = FWD_BASE; //- bias;
	  	      float cmdR = FWD_BASE; //+ bias;

	  	      // Safety caps — only enforce DUTY_MIN_MOVE when actually moving
	  	      if (!in_deadband) {
	  	          if (tc_dir_fb > 0) { // forward
	  	              if (cmdL < DUTY_MIN_MOVE) cmdL = DUTY_MIN_MOVE;
	  	              if (cmdR < DUTY_MIN_MOVE) cmdR = DUTY_MIN_MOVE;
	  	          } else {             // backward
	  	              if (cmdL > -DUTY_MIN_MOVE) cmdL = -DUTY_MIN_MOVE;
	  	              if (cmdR > -DUTY_MIN_MOVE) cmdR = -DUTY_MIN_MOVE;
	  	          }
	  	      } else {
	  	          cmdL = 0.f; cmdR = 0.f;                            // never creep in the band
	  	      }

	  	      if (cmdL >  DUTY_MAX_CAP)  cmdL =  DUTY_MAX_CAP;
	  	      if (cmdL < -DUTY_MAX_CAP)  cmdL = -DUTY_MAX_CAP;
	  	      if (cmdR >  DUTY_MAX_CAP)  cmdR =  DUTY_MAX_CAP;
	  	      if (cmdR < -DUTY_MAX_CAP)  cmdR = -DUTY_MAX_CAP;

	  	      if(err > 0){
	  	    	  MoveLeftWheel(cmdL);
	  	      }
	  	      else{
	  	    	  MoveRightWheel(cmdR);
	  	      }

//	  	      // --- arrival + stability ---
//	  	      if (in_deadband) {
//	  	          if (tc_stable_t0 == 0) tc_stable_t0 = now;
//	  	          if (now - tc_stable_t0 >= TC_STABLE_MS) {
//	  	              hardBrake();
//	  	              __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, SERVO_CENTER);
//	  	              tc_state = TC_IDLE;
//	  		          const char ack[] = "TWO";
//	  		          HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
//
//	  	          }
//	  	      } else {
//	  	          tc_stable_t0 = 0;
//	  	      }

	  	      // --- safety timeout ---
	  	      if (turnValue > 150 && now - tc_t0_ms > TC_TIMEOUT_MS2) {
	  	    	  hardBrake();
	  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, SERVO_CENTER);
	  	          tc_state = TC_IDLE;
	  	          const char ack[] = "AAAAA";
	  	          HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);

	  	      }
	  	      else if (turnValue < 150 && now - tc_t0_ms > TC_TIMEOUT_MS) {
	  	    	  	    	  hardBrake();
	  	    	  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, SERVO_CENTER);
	  	    	  	          tc_state = TC_IDLE;
	  	    	  	          const char ack[] = "AAAAA";
	  	    	  	          HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);

	  	    	  	      }

	  	      vTaskDelayUntil(&next, period);
	  	      continue;
	  	  }
	  else if (tc_state == MC_RUNNING)
	  	  {
	  	      int32_t pL = encA_pos, pR = encB_pos;
	  	      int32_t eL = mc_tL - pL;                  // error in counts
	  	      int32_t eR = mc_tR - pR;

	  	      float remL_cm = countsToCmL(llabs((long long)eL));
	  	      float remR_cm = countsToCmR(llabs((long long)eR));
	  	    int fb = directionayman ? +1 : -1;                 // forward=+1, backward=-1
	  	    	      float target = unwrap_towards(straightAngle, yaw_deg);
	  	    	      float err    = yaw_deg - target;              // shortest-way error (−180,180]

	  	    	      // update the servo using the correct sign
	  	    	      bool in_deadband = (fabsf(err) <= straightError);
	  	    	      if (in_deadband) {
	  	    	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);   // center
	  	    	      } else {
	  	    	          uint16_t servo_pwm = servo_from_err_fb3(err, fb);
	  	    	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
	  	    	      }

	  	      // inside MC_RUNNING (straight branch)
//	  	      int fb = direction ? +1 : -1;                 // forward=+1, backward=-1
//	  	      float err    = yaw_deg - target;              // shortest-way error (−180,180]

	  	      // update the servo using the correct sign
//	  	      bool in_deadband = (fabsf(err) <= straightError);
//	  	      if (in_deadband) {
//	  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);   // center
//	  	      } else {
//	  	          uint16_t servo_pwm = servo_from_err_fb2(err, fb);
//	  	          __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_pwm);
//	  	      }
	  //	      char msg[100];
	  //		  snprintf(msg, sizeof(msg), "yaw=%.2f, target=%.2f, err=%.2f\n",
	  //				   yaw_deg, straightAngle, err);
	  //		  HAL_UART_Transmit(&huart3, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);



	  //	      // ----- Hysteresis: enter at TOL_CM, exit only after TOL_CM + HYST_CM -----
	  //	      if (!L_in) L_in = (remL_cm <= TOL_CM);
	  //	      else       L_in = (remL_cm <= (TOL_CM + HYST_CM));
	  //
	  //	      if (!R_in) R_in = (remR_cm <= TOL_CM);
	  //	      else       R_in = (remR_cm <= (TOL_CM + HYST_CM));
	  	      //THESE ARE OUR MAIN STOPPING CONDITIONS
	  	      if (directionayman == true && (eL <= 0 || eR <= 0)){
	  	    	  hardBrake();
	  	    	  tc_state = TC_IDLE;
	  			  const char ack[] = "AAAAA";
	  			  HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
	  			  // (Optional) print rem distances here if you want

	  			  vTaskDelayUntil(&next, period);
	  			  continue;
	  	      }
	  	      if (directionayman == false && (eL >= 0 || eR >= 0)){
	  	    	  hardBrake();
	  	    	  tc_state = TC_IDLE;
	  			  const char ack[] = "AAAAA";
	  			  HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
	  			  // (Optional) print rem distances here if you want

	  			  vTaskDelayUntil(&next, period);
	  			  continue;
	  	      }


	  	      // ----- Taper speed near target (keeps momentum small) -----
	  	      float near_cm   = fminf(remL_cm, remR_cm);
	  	      //Absolute value
	  	      float cmRemaining = fabsf(near_cm);
	  	      float base_duty = 0.30f;
	  	      if (cmRemaining < 100.0f){
	  	    	  float s = cmRemaining/100.0f;
	  	    	  //, baseStraightSpeed * s * s * s
	  	    	  base_duty = fmaxf(0.099f, base_duty * s * s);
	  	      }



	  	      // ----- Drive only the wheel(s) still outside the window -----
	  	      float cmdL = 0.f, cmdR = 0.f;
	  	      if (!L_in) cmdL = ((eL > 0) ? +1 : -1) * base_duty;
	  	      if (!R_in) cmdR = ((eR > 0) ? +1 : -1) * base_duty;

	  	      MoveCar(cmdL, cmdR);

	  	      // Safety timeout (optional but recommended)
	  	      if ((HAL_GetTick() - mc_t0) > MC_TIMEOUT_MS) {
	  	          hardBrake();
	  	          tc_state = TC_IDLE;
	  	          const char ack[] = "AAAAA";
	  	          HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
	  	          vTaskDelayUntil(&next, period);
	  	          continue;
	  	      }

	  	      // Keep your periodic cadence even when running
	  	      vTaskDelayUntil(&next, period);
	  	  }

      // ===================== [ADD] WF_RUNNING state =====================
      // ===================== [ADD] WF_RUNNING state =====================
else if (tc_state == WF_RUNNING)
{
    // 1) Pick current side distance (raw)
    float d_wall_raw = (wf_side == WF_LEFT) ? g_ir_left_cm : g_ir_right_cm;

    // 2) Encoder-based distance since start
    int32_t dA = encA_pos - wf_encA0;
    int32_t dB = encB_pos - wf_encB0;
    float dA_cm = countsToCmL(llabs((long long)dA));
    float dB_cm = countsToCmR(llabs((long long)dB));
    wf_dist_cm  = 0.5f * (dA_cm + dB_cm);

    // NEW: keep "C" up to date while WL999/WR999 is active
    if (wf_curr_run_is_999) {
        wf_last_len_cm = wf_dist_cm;
    }

    // 3) Validate raw IR
    uint32_t now = HAL_GetTick();
    bool valid_raw = !(isnan(d_wall_raw) || isinf(d_wall_raw) ||
                       d_wall_raw < WF_MIN_CM || d_wall_raw > WF_MAX_CM);

    // 4) Jump rejection (single-sample spikes) + exponential smoothing
    bool valid = valid_raw;
    if (valid_raw && wf_prev_valid) {
        if (fabsf(d_wall_raw - wf_prev_cm) > wf_jump_cm) {
            valid = false; // reject this sample; keep previous filtered value
        }
    }
    if (valid) {
        wf_prev_cm    = wf_prev_valid ? (wf_alpha * d_wall_raw + (1.0f - wf_alpha) * wf_prev_cm)
                                      : d_wall_raw;
        wf_prev_valid = true;
    }

    // 5) Handle "lost" condition (noisy / out-of-range for a while)
    if (!valid) {
        if (wf_lost_since_ms == 0) wf_lost_since_ms = now;

        if (now - wf_lost_since_ms >= WF_LOST_HOLD_MS) {
            // 1) center steering first
            __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);
            vTaskDelay(pdMS_TO_TICKS(60));  // give servo ~60 ms to settle (tune 40–120)

            // 2) then stop + report measured length
            hardBrake();
            tc_state = TC_IDLE;

            // Save C only for WL999/WR999 runs
            if (wf_curr_run_is_999) {
                wf_last_len_cm = wf_dist_cm;     // <- C in cm
                wf_curr_run_is_999 = false;      // clear the flag after capture
            }

            char buf[40];
//            int n = snprintf(buf, sizeof buf, "WD%.1f\r\n", (double)wf_dist_cm);
//            HAL_UART_Transmit(&huart3, (uint8_t*)buf, n, HAL_MAX_DELAY);

            const char ack[] = "AAAAA";
            HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
            vTaskDelayUntil(&next, period);
            continue;
        }

        // brief glitch: gentle straight cruise while we wait (local slower)
        __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servoValue);
        float cruise = 0.12f;  // softer than wf_base_duty, local to WF
        float l = slew_f(wf_cmdL_prev, cruise, 0.012f);
        float r = slew_f(wf_cmdR_prev, cruise, 0.012f);
        wf_cmdL_prev = l; wf_cmdR_prev = r;
        MoveCar(l, r);
        vTaskDelayUntil(&next, period);
        continue;
    } else {
        wf_lost_since_ms = 0;
    }

    // 6) Lock-to-current setpoint (once stable)
    if (wf_lock_pending) {
        if (valid) {
            if (wf_lock_good_cnt < 255) wf_lock_good_cnt++;
        } else {
            wf_lock_good_cnt = 0;
        }
        bool timed_out = (now - wf_lock_t0_ms) >= WF_LOCK_ACQUIRE_MS;
        if ((wf_lock_good_cnt >= 3) || timed_out) {
            float lock_cm = wf_prev_cm;      // filtered value
            if (!isfinite(lock_cm) || lock_cm < WF_MIN_CM || lock_cm > WF_MAX_CM) {
                lock_cm = 20.0f;             // fallback
            }
            wf_set_cm = round_cm(lock_cm);
            wf_lock_last_set = wf_set_cm;
            wf_lock_pending = false;

            // (optional) announce lock
            char msg[24];
//            int n = snprintf(msg, sizeof msg, "WLOCK%.0f\r\n", (double)wf_set_cm);
//            HAL_UART_Transmit(&huart3, (uint8_t*)msg, n, HAL_MAX_DELAY);
        }
    }

    // 7) Control to setpoint using FILTERED distance
    float d_wall = wf_prev_cm; // filtered distance
    float err_cm = (wf_set_cm - d_wall) * ((wf_side == WF_LEFT) ? +1.0f : -1.0f);

    /* ---------------- slow profile + near-setpoint damping (uses globals) --------------- */
    float abs_e = fabsf(err_cm);

    // --- Hysteresis update (enter/leave acceptance band) ---
    if (!wf_ok_active && abs_e <= wf_ok_in_cm) {
        wf_ok_active = true;     // good enough: stop turning, cruise straight
    } else if (wf_ok_active && abs_e >= wf_ok_out_cm) {
        wf_ok_active = false;    // error grew again: allow turning
    }


    // Forward speed taper around setpoint
    float taper   = clampf(abs_e / wf_slow_window_cm, 0.f, 1.f);
    float base_fw = wf_min_slow + taper * (wf_base_slow - wf_min_slow);

    // Gain schedule near setpoint (soften when |err| small)
    float kp_servo_eff = (abs_e < wf_near_cm) ? (wf_kp_servo_near * wf_kp_servo) : wf_kp_servo;
    float kp_diff_eff  = (abs_e < wf_near_cm) ? (wf_kp_diff_near  * wf_kp_diff ) : wf_kp_diff;

    // Tiny deadband for servo to remove jitter
    if (abs_e <= wf_servo_db_cm) kp_servo_eff = 0.0f;

    // If inside acceptance band: go straight (no steering, no L/R bias)
    if (wf_ok_active) {
        kp_servo_eff = 0.0f;                 // centers servo (via existing formula)
        kp_diff_eff  = 0.0f;                 // zero wheel bias
        base_fw = fmaxf(base_fw, wf_ok_min_fw); // keep a bit of straight cruise
    }

    // Tighter slews when close
    uint16_t servo_slew_step = (abs_e < wf_near_cm) ? wf_servo_slew_near : wf_servo_slew;
    float    duty_slew_step  = (abs_e < wf_near_cm) ? wf_duty_slew_near  : wf_duty_slew;

    /* ---------------- servo with slew limit ---------------- */
    float servo_pwm_f = SERVO_CENTER + kp_servo_eff * err_cm;
    if (servo_pwm_f < SERVO_PWM_MIN) servo_pwm_f = SERVO_PWM_MIN;
    if (servo_pwm_f > SERVO_PWM_MAX) servo_pwm_f = SERVO_PWM_MAX;
    uint16_t servo_target = (uint16_t)servo_pwm_f;
    uint16_t servo_cmd    = slew_u16(wf_servo_prev, servo_target, servo_slew_step);
    wf_servo_prev = servo_cmd;
    __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, servo_cmd);

    /* ---------------- wheel bias with slew + smaller clamps ---------------- */
    float diff = clampf(kp_diff_eff * err_cm, -wf_diff_clamp, wf_diff_clamp);
    float cmdL_target = clampf(base_fw + ((wf_side==WF_LEFT) ? +diff : -diff), -0.6f, 0.6f);
    float cmdR_target = clampf(base_fw - ((wf_side==WF_LEFT) ? +diff : -diff), -0.6f, 0.6f);

    float cmdL = slew_f(wf_cmdL_prev, cmdL_target, duty_slew_step);
    float cmdR = slew_f(wf_cmdR_prev, cmdR_target, duty_slew_step);
    wf_cmdL_prev = cmdL; wf_cmdR_prev = cmdR;

    MoveCar(cmdL, cmdR);



    vTaskDelayUntil(&next, period);
    continue;
}

	  else if (tc_state == TC_IDLE){
		  hardBrake();
	  }
      vTaskDelayUntil(&next, period);           // <- fix: yield so UltraTask can update distance

  }
  /* USER CODE END MoveCarTask */
}

/* USER CODE BEGIN Header_EncoderTask */
/**
* @brief Function implementing the encoderTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_EncoderTask */
void EncoderTask(void *argument)
{
  /* USER CODE BEGIN EncoderTask */
	TickType_t next = xTaskGetTickCount();
	const TickType_t period = pdMS_TO_TICKS(10);
  /* Infinite loop */
  for(;;)
  {
	  vTaskDelayUntil(&next, period);
	      SampleEncoder(0.010f);
  }
  /* USER CODE END EncoderTask */
}

/* USER CODE BEGIN Header_OledTask */
/**
* @brief Function implementing the oledTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_OledTask */
void OledTask(void *argument)
{
  /* USER CODE BEGIN OledTask */
	  OLED_Init();
	  OLED_Clear();

	  for (;;)
	  {
	    // Copy the volatile globals into locals so we print a consistent snapshot
	    uint32_t echo     = echo_us;
	    float    dist_cm  = distance_cm;
	    float    irL_cm   = g_ir_left_cm;   // PA3
	    float    irR_cm   = g_ir_right_cm;  // PA2


	    char line1[24], line2[24], line3[24], line4[24];

	    // Format lines (1 per row)
	    // Line 1: Echo time
	    snprintf(line1, sizeof(line1), "%.2f, %.2f", differenceobject1, differenceobject2);

	    // Line 2: Ultrasonic distance
//	    if (dist_cm <= 0.01f) snprintf(line2, sizeof(line2), "Dist: --.- cm");
//	    else                  snprintf(line2, sizeof(line2), "Dist: %.1f cm", (double)dist_cm);
	    snprintf(line2, sizeof(line2), "Dumee: %.2f", Xf);

	    // Line 3: IR Left (PA3)
	    if (irL_cm <= 0.01f) snprintf(line3, sizeof(line3), "IR L: --.- cm");
	    else                 snprintf(line3, sizeof(line3), "IR L: %.1f cm", (double)irL_cm);

	    // Line 4: IR Right (PA2)
	    if (irR_cm <= 0.01f) snprintf(line4, sizeof(line4), "IR R: --.- cm");
	    else                 snprintf(line4, sizeof(line4), "IR R: %.1f cm", (double)irR_cm);

	    // Draw all four lines
	    OLED_Clear();
	    OLED_ShowString(0,  0, (uint8_t*)line1);
	    OLED_ShowString(0, 16, (uint8_t*)line2);
	    OLED_ShowString(0, 32, (uint8_t*)line3);
	    OLED_ShowString(0, 48, (uint8_t*)line4);
	    OLED_Refresh_Gram();

	    osDelay(200);   // ~10 Hz refresh
	  }
  /* USER CODE END OledTask */
}

/* USER CODE BEGIN Header_UltraTask */
/**
* @brief Function implementing the ultraTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_UltraTask */
void UltraTask(void *argument)
{
  /* USER CODE BEGIN UltraTask */
//	OLED_Init();
	  const TickType_t period = pdMS_TO_TICKS(10);
	  TickType_t next = xTaskGetTickCount();

	  for (;;)
	  {
	    // 10 µs TRIG pulse on PB14
	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);
	    vTaskDelay(pdMS_TO_TICKS(1));    // small settle
	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);
	    __HAL_TIM_SET_COUNTER(&htim6, 0);
	    while (__HAL_TIM_GET_COUNTER(&htim6) < 10) {}   // us_delay(10)
	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);

	    if (ic_captured) {
	      ic_captured = 0;

	      // Distance in 0.1 cm: (echo_us * 343) / 2000
	      uint32_t tenths_cm = (echo_us * 343u) / 2000u;
	      uint32_t cm_whole  = tenths_cm / 10u;
	      uint32_t cm_frac1  = tenths_cm % 10u;
	      distance_cm = (float)tenths_cm * 0.1f;          // -> cm as float


//	      char line1[24], line2[24], line3[24], line4[24];
//	      snprintf(line1, sizeof(line1), "Echo: %lu us ", (unsigned long)echo_us);
//
//	      //snprintf(line1, sizeof(line1), "MYDIST: %.1f cm", (double)distance_cm);
//	      snprintf(line2, sizeof(line2), "Dist: %lu.%lu cm",
//	               (unsigned long)cm_whole, (unsigned long)cm_frac1);
//
//	      // Line 3: IR Left (PA3)
//	   	  snprintf(line3, sizeof(line3), "IR L: %.1f cm", (double)g_ir_left_cm);
//
//	   	  // Line 4: IR Right (PA2)
//	   	  snprintf(line4, sizeof(line4), "IR R: %.1f cm", (double)g_ir_right_cm);
//
//
//	      OLED_Clear();
//	      OLED_ShowString(0,  0, (uint8_t*)line1);
//	      OLED_ShowString(0, 16, (uint8_t*)line2);
//	      OLED_Refresh_Gram();
	    }

	    vTaskDelayUntil(&next, period);
	  }
  /* USER CODE END UltraTask */
}

/* USER CODE BEGIN Header_CommunicateTask */
/**
* @brief Function implementing the communicateTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_CommunicateTask */
void CommunicateTask(void *argument)
{
  /* USER CODE BEGIN CommunicateTask */
	const uint8_t ack[5] = {'A', 'A', 'A', 'A', 'A'};

  /* Infinite loop */
  for(;;)
  {
	  GPIO_PinState s = HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_0);
	  if (s == GPIO_PIN_RESET){
		  	  //LEFT
			//start_arc_turn('R', 'F', 90);
		  tc_state = Phase1;

		}
	  if (rxReady){
	      	rxReady = 0; //Consume frame

	      	//Read the 5 byte data
	  		uint8_t c0 = aRxBuffer[0];
	  		uint8_t c1 = aRxBuffer[1];
	  		uint8_t d2 = aRxBuffer[2];
	  		uint8_t d3 = aRxBuffer[3];
	  		uint8_t d4 = aRxBuffer[4];

	  		if (c0=='S' && c1=='T' && d2=='A' && d3=='R' && d4=='T'){
	  			taskENTER_CRITICAL();
	  			straightAngle = yaw_deg;
	  			taskEXIT_CRITICAL();
	  			if(numObstacle == 1){
	  				object1dleft = encA_pos;
	  				object1dright = encB_pos;
	  			}
	  			else{
	  				object2dleft = encA_pos;
	  				object2dright = encB_pos;
	  			}
	  			directionayman = true;
	  			tc_state = Phase1;
	  		}
	  		else if (c0=='S' && c1=='N' && d2=='A' && d3=='P' && d4=='P'){
	  			snap();
			  const char ack[] = "AAAAA";
			  HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);

//			  char buffer[30];
//			  int len = sprintf(buffer, "%5.2f, %5.2f\r\n", differenceobject1, differenceobject2);
//			  HAL_UART_Transmit(&huart3, (const uint8_t*)buffer, (uint16_t)len, HAL_MAX_DELAY);

	  		}
	  		else if (c0=='C' && c1=='O' && d2=='R' && d3=='R' && d4=='E'){
	  			tc_state = CORRECTION;
	  		}
	  		else if (c0=='E' && c1=='N' && d2=='Y' && d3=='A' && d4=='O'){
	  			  			tc_state = CORRECTION2;
	  			  		}
	  		else if (c0=='L' && c1=='F' && is_digit(d2) && is_digit(d3) && is_digit(d4)){
				int mag = parse_magnitude(&aRxBuffer[2]);
				turnValue = parse_magnitude(&aRxBuffer[2]);
	  			direction = 2;
	  			start_arc_turn('R', 'F', (float)mag);
	  		}

	  		else if (c0=='R' && c1=='F' && is_digit(d2) && is_digit(d3) && is_digit(d4)){
				int mag = parse_magnitude(&aRxBuffer[2]);
				turnValue = parse_magnitude(&aRxBuffer[2]);
	  			direction = 1;
	  			start_arc_turn('L', 'F', (float)mag);
	  		}


	  		else if (c0=='S' && is_digit(d2) && is_digit(d3) && is_digit(d4)){
	  			int mag = parse_magnitude(&aRxBuffer[2]);
	  			if (c1 == 'B'){
					mag = -mag;
					directionayman = false;
				}
	  			else{
	  				directionayman = true;
	  			}

				start_move_straight((float)mag, 0.0f);
	  		}

else if (c0=='W' && (c1=='L' || c1=='R')) {
    // WLddd / WRddd  OR  WL999 / WR999 => lock-to-current
    if (d2=='9' && d3=='9' && d4=='9') {
        start_wall_follow((c1=='L') ? WF_LEFT : WF_RIGHT, -1.0f, 0.25f);  // lock mode
    } else if (is_digit(d2) && is_digit(d3) && is_digit(d4)) {
        int mag = parse_magnitude(&aRxBuffer[2]);   // desired setpoint in cm
        if (mag < 10) mag = 10; if (mag > 60) mag = 60;
        start_wall_follow((c1=='L') ? WF_LEFT : WF_RIGHT, (float)mag, 0.25f);
    }
}
else if (c0=='H' && c1=='O' && d2=='M' && d3=='E' && d4=='E') {
    RunHOMEE();
//    const char ack[] = "AAAAA";
//    HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
}
else if (c0=='D' && c1=='U' && d2=='M' && d3=='E' && d4=='E') {
    RunDUMEE();
//    const char ack[] = "AAAAA";
//    HAL_UART_Transmit(&huart3, (uint8_t*)ack, sizeof(ack)-1, HAL_MAX_DELAY);
}

	  }

	  osDelay(5);
  }
  /* USER CODE END CommunicateTask */
}

/* USER CODE BEGIN Header_GyroTask */
/**
* @brief Function implementing the gyroTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_GyroTask */
void GyroTask(void *argument)
{
  /* USER CODE BEGIN GyroTask */
	// Your helpers already set up AUX I2C + mag in ICM_Initialize()
	  ICM_lock();
	  ICM_SelectBank(0);
	  HAL_Delay(5);
	  (void)ICM_WHOAMI();

	  // Do the same sequence you had, but ensure ICM_Initialize() runs once:
	  ICM_SetClock(0x01);          // auto clock / wake
	  osDelay(5);
	  ICM_AccelGyroOff();
	  osDelay(5);
	  ICM_AccelGyroOn();
	  osDelay(5);
	  ICM_SelectBank(2);
	  osDelay(5);
	  ICM_SetGyroRateLPF(GYRO_RATE_250, GYRO_LPF_17HZ);
	  osDelay(10);

	  // Make sure full initialization (incl. mag path) is done:
	  (void)ICM_Initialize();      // sets sample rates + AUX I2C + mag

	  // Back to bank 0 for data reads
	  ICM_SelectBank(0);
	  ICM_unlock();

	  osDelay(20);
	  {
		  uint32_t t0 = osKernelGetTickCount();
		  int32_t acc = 0, n = 0;
		  uint8_t raw[12];
		  while (osKernelGetTickCount() - t0 < 4000) {
			if (ICM_readBytes(0x2D, raw, 12) == HAL_OK) {
			  int16_t gzr = (int16_t)((raw[10] << 8) | raw[11]);
			  acc += gzr; n++;
			}
			osDelay(5);
		  }
		  if (n > 0) gz_bias = (float)acc / (float)n;
		}
	  uint32_t t_prev = osKernelGetTickCount();
	  uint32_t last_mag_ms = t_prev;
  /* Infinite loop */
  for(;;)
  {
	  // dt from RTOS tick (ms -> s)
	  uint32_t t_now = osKernelGetTickCount();
	  float dt = (t_now - t_prev) * 0.001f; if (dt <= 0) dt = 0.001f;
	  t_prev = t_now;

	  // --- Read gyro Z only ---
	  uint8_t raw[12];
	  if (ICM_readBytes(0x2D, raw, 12) == HAL_OK) {
		int16_t gzr = (int16_t)((raw[10] << 8) | raw[11]);
		float gz_dps = (gzr - gz_bias) / GYRO_LSB_PER_DPS;

		// integrate gyro Z
		float yaw_pred = yaw_deg + gz_dps * dt;  // in degrees
		yaw_deg = wrap360(yaw_pred);
	  }

	  // --- Occasionally read mag and correct (e.g. every 100 ms) ---
	  if (tc_state == TC_IDLE && (t_now - last_mag_ms) >= 80) {
		  last_mag_ms = t_now;
		  yaw_alpha = 0.99f;
	  }
	  else {
		  yaw_alpha = 1.0f;
	  }
		int16_t mraw[3];
		if (ICM_ReadMag(mraw) == HAL_OK) {
		  // apply simple hard/soft-iron
		  float mx = (mraw[0] - mag_bx) * mag_sx;
		  float my = (mraw[1] - mag_by) * mag_sy;
		  float mz = (mraw[2] - mag_bz) * mag_sz;
		  (void)mz; // not used if we skip tilt comp

		  // assume robot is near-level → 2D heading is OK
		  float mag_heading = atan2f(-my, mx) * 57.2957795f; // [-180,180]
		  if (mag_heading < 0) mag_heading += 360.0f;


		  if (!yaw_inited) {
			  yaw_deg = mag_heading;
			  yaw_inited = 1;
		  }
		  else {
			// unwrap target towards gyro yaw, then blend
			float mag_unwrapped = unwrap_towards(mag_heading, yaw_deg);
			float blended = yaw_alpha * yaw_deg + (1.0f - yaw_alpha) * mag_unwrapped;
			yaw_deg = wrap360(blended);
		  }
		}

	  // --- publish / print if you want ---
	  // total_angle = yaw_deg;  // share to rest of app
	  // printf or UART…
//	      char buffer[30];
//	      int len = sprintf(buffer, "%5.2f\r\n", yaw_deg);
//	      HAL_UART_Transmit(&huart3, (const uint8_t*)buffer, (uint16_t)len, HAL_MAX_DELAY);


	  osDelay(5); // ~200 Hz gyro integration, mag at 10 Hz
  }
  /* USER CODE END GyroTask */
}

/* USER CODE BEGIN Header_InfraTask */
/**
* @brief Function implementing the infraTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_InfraTask */
void InfraTask(void *argument)
{
  /* USER CODE BEGIN InfraTask */
	  // ---- Config ----
	  #define FILTER_N 5
	  const float VREF = 3.3f;

	  // ---- State ----
	  static uint16_t ring1[FILTER_N] = {0};
	  static uint16_t ring2[FILTER_N] = {0};
	  static uint8_t  idx = 0, filled = 0;

	  for (;;)
	  {
	    uint16_t raw1 = 0, raw2 = 0;

	    // Try to start the 2-rank scan (Rank1=CH2, Rank2=CH3 are set in MX_ADC1_Init)
	    // If ADC is busy for some reason, back off and retry next tick.
	    if (HAL_ADC_Start(&hadc1) != HAL_OK) {
	      HAL_ADC_Stop(&hadc1);
	      osDelay(2);
	      continue;
	    }

	    // Read Rank 1 -> CH2 (PA2)
	    if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK) {
	      raw1 = HAL_ADC_GetValue(&hadc1);
	    } else {
	      HAL_ADC_Stop(&hadc1);
	      osDelay(2);
	      continue;  // recover next loop
	    }

	    // Read Rank 2 -> CH3 (PA3)
	    if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK) {
	      raw2 = HAL_ADC_GetValue(&hadc1);
	    } else {
	      HAL_ADC_Stop(&hadc1);
	      osDelay(2);
	      continue;  // recover next loop
	    }

	    HAL_ADC_Stop(&hadc1);

	    // Moving average
	    ring1[idx] = raw1;
	    ring2[idx] = raw2;
	    idx++;
	    if (idx >= FILTER_N) { idx = 0; filled = 1; }

	    uint8_t  n = filled ? FILTER_N : idx;
	    uint32_t s1 = 0, s2 = 0;
	    for (uint8_t i = 0; i < n; i++) { s1 += ring1[i]; s2 += ring2[i]; }

	    uint16_t avg1 = (n ? (uint16_t)(s1 / n) : raw1); // PA2
	    uint16_t avg2 = (n ? (uint16_t)(s2 / n) : raw2); // PA3

	    // Convert to voltages (guard rails to avoid powf(0, negative))
	    float v1 = VREF * (float)avg1 / 4095.0f;  // PA2 (Right)
	    float v2 = VREF * (float)avg2 / 4095.0f;  // PA3 (Left)
	    if (v1 < 0.05f) v1 = 0.05f;
	    if (v2 < 0.05f) v2 = 0.05f;

	    // Distance curve (placeholder; tune with your calibration)
	    // d ≈ k * v^p  (k=27.728, p=−1.2045 from earlier)
	    float d_right = 27.728f * powf(v1, -1.2045f);  // PA2
	    float d_left  = 27.728f * powf(v2, -1.2045f);  // PA3

	    // Publish to globals (32-bit writes are atomic on Cortex-M4)
	    g_ir_right_cm = d_right;  // PA2
	    g_ir_left_cm  = d_left;   // PA3

//	     --- DEBUG UART (re-enable after stable) ---
//	     char msg[120];
//	     int len = snprintf(msg, sizeof(msg),
//	       "IRR: RAW=%4u V=%.2f D=%.1f | IRL: RAW=%4u V=%.2f D=%.1f\r\n",
//	        avg1, v1, d_right,       avg2, v2, d_left);
//	     HAL_UART_Transmit(&huart3, (uint8_t*)msg, len, 50);

	    osDelay(50);  // ~20 Hz. Don’t make this 0; keep the system breathable.
	  }
  /* USER CODE END InfraTask */
}

/**
  * @brief  Period elapsed callback in non blocking mode
  * @note   This function is called  when TIM7 interrupt took place, inside
  * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
  * a global variable "uwTick" used as application time base.
  * @param  htim : TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM7)
  {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
