#include <Arduino.h>
#include <ESP32Servo.h>

static const int SERVO_PIN = 15;
static const int SERVO_LEFT_ANGLE = 30;
static const int SERVO_CENTER_ANGLE = 73;
static const int SERVO_RIGHT_ANGLE = 150;
static const unsigned long SERVO_HOLD_MS = 2000;
static const unsigned long SERVO_COOLDOWN_MS = 1000;

Servo sorterServo;
String incomingCommand;
unsigned long returnCenterAt = 0;
unsigned long nextCommandAllowedAt = 0;

void moveServo(int angle) {
  delay(2000);
  sorterServo.write(angle);
}

void queueReturnCenter() {
  returnCenterAt = millis() + SERVO_HOLD_MS;
  nextCommandAllowedAt = returnCenterAt + SERVO_COOLDOWN_MS;
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (millis() < nextCommandAllowedAt) {
    Serial.println("ERR BUSY");
    return;
  }

  if (command == "LEFT" || command == "RECYCLE") {
    moveServo(SERVO_LEFT_ANGLE);
    queueReturnCenter();
    Serial.println("ACK LEFT");
    return;
  }

  if (command == "RIGHT" || command == "TRASH") {
    moveServo(SERVO_RIGHT_ANGLE);
    queueReturnCenter();
    Serial.println("ACK RIGHT");
    return;
  }

  if (command == "CENTER") {
    moveServo(SERVO_CENTER_ANGLE);
    returnCenterAt = 0;
    Serial.println("ACK CENTER");
    return;
  }

  if (command.length() > 0) {
    Serial.print("ERR UNKNOWN COMMAND: ");
    Serial.println(command);
  }
}

void setup() {
  Serial.begin(115200);
  sorterServo.setPeriodHertz(50);
  sorterServo.attach(SERVO_PIN, 500, 2400);
  moveServo(SERVO_CENTER_ANGLE);
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n' || c == '\r') {
      if (incomingCommand.length() > 0) {
        handleCommand(incomingCommand);
        incomingCommand = "";
      }
      continue;
    }

    incomingCommand += c;
  }

  if (returnCenterAt != 0 && millis() >= returnCenterAt) {
    sorterServo.write(SERVO_CENTER_ANGLE);
    returnCenterAt = 0;
    Serial.println("ACK CENTER");
  }
}