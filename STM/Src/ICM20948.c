/*
 * ICM20948.c
 *
 *  Created on: Oct 26, 2018
 *      Author: cory
 */

// *** Three asterisks to the side of a line means this may change based on platform
#include "main.h" // ***
extern I2C_HandleTypeDef hi2c2;   // <- tell this file the handle lives in main.c

#include "ICM20948.h"
#include <string.h>
#include "cmsis_os2.h"
static osMutexId_t s_icm_mutex = NULL;



#ifndef ICM_ADDR_7B
#define ICM_ADDR_7B   0x68          // 0x69 if AD0=HIGH
#endif
#define ICM_ADDR      (ICM_ADDR_7B << 1)

void ICM_LockInit(void) {
    if (s_icm_mutex == NULL) {
        s_icm_mutex = osMutexNew(NULL);
    }
}

void ICM_lock(void) {
    if (s_icm_mutex) osMutexAcquire(s_icm_mutex, osWaitForever);
}

void ICM_unlock(void) {
    if (s_icm_mutex) osMutexRelease(s_icm_mutex);
}

// ---- I2C primitives ----
HAL_StatusTypeDef ICM_WriteOneByte(uint8_t reg, uint8_t val) {
    return HAL_I2C_Mem_Write(&hi2c2, ICM_ADDR, reg, I2C_MEMADD_SIZE_8BIT, &val, 1, 100);
}

HAL_StatusTypeDef ICM_ReadOneByte(uint8_t reg, uint8_t *val) {
    return HAL_I2C_Mem_Read(&hi2c2, ICM_ADDR, reg, I2C_MEMADD_SIZE_8BIT, val, 1, 100);
}


HAL_StatusTypeDef ICM_readBytes(uint8_t start, uint8_t *p, uint16_t n){
  ICM_lock();
  ICM_SelectBank(USER_BANK_0);
  HAL_StatusTypeDef s = HAL_I2C_Mem_Read(&hi2c2, ICM_ADDR, start, I2C_MEMADD_SIZE_8BIT, p, n, 100);
  ICM_unlock();
  return s;
}

void ICM_Mag_ReadBurst(uint8_t reg, uint8_t *dst, uint8_t n){
  ICM_lock();
  ICM_SelectBank(USER_BANK_3);
  ICM_WriteOneByte(0x03, 0x8C);      // read mode
  ICM_WriteOneByte(0x04, reg);
  ICM_WriteOneByte(0x06, 0xFF);      // trigger
  ICM_SelectBank(USER_BANK_0);
  for(uint8_t i=0;i<n;i++) ICM_ReadOneByte(0x3B+i, &dst[i]); // EXT_SENS_DATA
  ICM_unlock();
}


// ---- SPI-only leftovers become no-ops on I2C ----
void ICM_CSHigh(void) { /* no CS in I2C */ }
void ICM_CSLow(void)  { /* no CS in I2C */ }


/*
 *
 * AUX I2C abstraction for magnetometer
 *
 */
void i2c_Mag_write(uint8_t reg,uint8_t value)
  {
  	ICM_WriteOneByte(0x7F, 0x30);

  	HAL_Delay(1);
  	ICM_WriteOneByte(0x03 ,0x0C);//mode: write

  	HAL_Delay(1);
  	ICM_WriteOneByte(0x04 ,reg);//set reg addr

  	HAL_Delay(1);
  	ICM_WriteOneByte(0x06 ,value);//send value

  	HAL_Delay(1);
  }

  static uint8_t ICM_Mag_Read(uint8_t reg)
  {
  	uint8_t  Data;
  	ICM_WriteOneByte(0x7F, 0x30);
    HAL_Delay(1);
  	ICM_WriteOneByte(0x03 ,0x0C|0x80);
    HAL_Delay(1);
  	ICM_WriteOneByte(0x04 ,reg);// set reg addr
    HAL_Delay(1);
  	ICM_WriteOneByte(0x06 ,0xff);//read
  	HAL_Delay(1);
  	ICM_WriteOneByte(0x7F, 0x00);
  	ICM_ReadOneByte(0x3B,&Data);
    HAL_Delay(1);
  	return Data;
  }

  void ICM20948_READ_MAG(int16_t magn[3])
  {
    uint8_t mag_buffer[10];

      mag_buffer[0] =ICM_Mag_Read(0x01);

      mag_buffer[1] =ICM_Mag_Read(0x11);
  	  mag_buffer[2] =ICM_Mag_Read(0x12);
  	  magn[0]=mag_buffer[1]|mag_buffer[2]<<8;
    	mag_buffer[3] =ICM_Mag_Read(0x13);
      mag_buffer[4] =ICM_Mag_Read(0x14);
    	magn[1]=mag_buffer[3]|mag_buffer[4]<<8;
  	 	mag_buffer[5] =ICM_Mag_Read(0x15);
      mag_buffer[6] =ICM_Mag_Read(0x16);
  		magn[2]=mag_buffer[5]|mag_buffer[6]<<8;

     	i2c_Mag_write(0x31,0x01);
  }

/*
 *
 * Read magnetometer
 *
 */
void ICM_ReadMag(int16_t magn[3]) {
	uint8_t mag_buffer[10];

	      mag_buffer[0] =ICM_Mag_Read(0x01);

	      mag_buffer[1] =ICM_Mag_Read(0x11);
	  	  mag_buffer[2] =ICM_Mag_Read(0x12);
	  	  magn[0]=mag_buffer[1]|mag_buffer[2]<<8;
	    	mag_buffer[3] =ICM_Mag_Read(0x13);
	      mag_buffer[4] =ICM_Mag_Read(0x14);
	    	magn[1]=mag_buffer[3]|mag_buffer[4]<<8;
	  	 	mag_buffer[5] =ICM_Mag_Read(0x15);
	      mag_buffer[6] =ICM_Mag_Read(0x16);
	  		magn[2]=mag_buffer[5]|mag_buffer[6]<<8;

	     	i2c_Mag_write(0x31,0x01);
}

/*
 *
 * Sequence to setup ICM290948 as early as possible after power on
 *
 */
void ICM_PowerOn(void) {
	char uart_buffer[200];
	uint8_t whoami = 0xEA;
	uint8_t test = ICM_WHOAMI();
	//if (test == whoami) {
		ICM_CSHigh();
		HAL_Delay(10);
		ICM_SelectBank(USER_BANK_0);
		HAL_Delay(10);

		HAL_Delay(10);
		ICM_SetClock((uint8_t)CLK_BEST_AVAIL);
		HAL_Delay(10);
		ICM_AccelGyroOff();
		HAL_Delay(20);
		ICM_AccelGyroOn();
		HAL_Delay(10);
		ICM_Initialize();
	//} else {
		//sprintf(uart_buffer, "Failed WHO_AM_I.  %i is not 0xEA\r\n", test);
		//HAL_UART_Transmit_DMA(UART_BUS, (uint8_t*) uart_buffer, strlen(uart_buffer));
		//HAL_Delay(100);
	//}
}
uint16_t ICM_Initialize(void) {
		ICM_SelectBank(USER_BANK_2);
		HAL_Delay(20);
		ICM_SetGyroRateLPF(GYRO_RATE_250, GYRO_LPF_17HZ);
		HAL_Delay(10);

		// Set gyroscope sample rate to 100hz (0x0A) in GYRO_SMPLRT_DIV register (0x00)
		ICM_WriteOneByte(0x00, 0x0A);
		HAL_Delay(10);

		// Set accelerometer low pass filter to 136hz (0x11) and the rate to 8G (0x04) in register ACCEL_CONFIG (0x14)
		ICM_WriteOneByte(0x14, (0x04 | 0x11));

		// Set accelerometer sample rate to 225hz (0x00) in ACCEL_SMPLRT_DIV_1 register (0x10)
		ICM_WriteOneByte(0x10, 0x00);
		HAL_Delay(10);

		// Set accelerometer sample rate to 100 hz (0x0A) in ACCEL_SMPLRT_DIV_2 register (0x11)
		ICM_WriteOneByte(0x11, 0x0A);
		HAL_Delay(10);

		ICM_SelectBank(USER_BANK_2);
		HAL_Delay(20);

		// Configure AUX_I2C Magnetometer (onboard ICM-20948)
		ICM_WriteOneByte(0x7F, 0x00); // Select user bank 0
		ICM_WriteOneByte(0x0F, 0x30); // INT Pin / Bypass Enable Configuration
		ICM_WriteOneByte(0x03, 0x20); // I2C_MST_EN
		ICM_WriteOneByte(0x7F, 0x30); // Select user bank 3
		ICM_WriteOneByte(0x01, 0x4D); // I2C Master mode and Speed 400 kHz
		ICM_WriteOneByte(0x02, 0x01); // I2C_SLV0 _DLY_ enable
		ICM_WriteOneByte(0x05, 0x81); // enable IIC	and EXT_SENS_DATA==1 Byte

		// Initialize magnetometer
		i2c_Mag_write(0x32, 0x01); // Reset AK8963
		HAL_Delay(1000);
		i2c_Mag_write(0x31, 0x02); // use i2c to set AK8963 working on Continuous measurement mode1 & 16-bit output

		return 1337;
	}

void ICM_ReadAccelGyro(void) {
	uint8_t raw_data[12];
	ICM_readBytes(0x2D, raw_data, 12);

	accel_data[0] = (raw_data[0] << 8) | raw_data[1];
	accel_data[1] = (raw_data[2] << 8) | raw_data[3];
	accel_data[2] = (raw_data[4] << 8) | raw_data[5];

	gyro_data[0] = (raw_data[6] << 8) | raw_data[7];
	gyro_data[1] = (raw_data[8] << 8) | raw_data[9];
	gyro_data[2] = (raw_data[10] << 8) | raw_data[11];

	accel_data[0] = accel_data[0] / 8;
	accel_data[1] = accel_data[1] / 8;
	accel_data[2] = accel_data[2] / 8;

	gyro_data[0] = gyro_data[0] / 250;
	gyro_data[1] = gyro_data[1] / 250;
	gyro_data[2] = gyro_data[2] / 250;
}
void ICM_SelectBank(uint8_t bank) {
	ICM_WriteOneByte(USER_BANK_SEL, bank);
}
void ICM_Disable_I2C(void) {
	ICM_WriteOneByte(0x03, 0x78);
}

void ICM_SetClock(uint8_t clk) {
	ICM_WriteOneByte(PWR_MGMT_1, clk);
}
void ICM_AccelGyroOff(void) {
	ICM_WriteOneByte(PWR_MGMT_2, (0x38 | 0x07));
}
void ICM_AccelGyroOn(void) {
	ICM_WriteOneByte(0x07, (0x00 | 0x00));
}
uint8_t ICM_WHOAMI(void) {
	uint8_t spiData = 0x01;
	ICM_ReadOneByte(0x00, &spiData);
	return spiData;
}
void ICM_SetGyroRateLPF(uint8_t rate, uint8_t lpf) {
	ICM_WriteOneByte(GYRO_CONFIG_1, (rate|lpf));
}
/*
 *
 * Read Accelerometer and Gyro data
 *
 */
