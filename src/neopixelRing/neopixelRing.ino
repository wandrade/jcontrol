#include <Adafruit_NeoPixel.h>
#include <Wire.h>

#define PIN 9
 
// Parameter 1 = number of pixels in strip
// Parameter 2 = pin number (most are valid)
// Parameter 3 = pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
Adafruit_NeoPixel strip = Adafruit_NeoPixel(16, PIN, NEO_GRB + NEO_KHZ800);

int mode = 0;
byte r = 16;
byte g = 123;
byte b = 65;
byte waitTime = 80;
byte flag = 0;
byte brightness = 30;
byte c[10];

void setup() {
  strip.begin();
  strip.setBrightness(brightness); //adjust brightness here
  strip.show(); // Initialize all pixels to 'off'
  Wire.begin(0x55);                // join i2c bus with address #8
  Wire.onReceive(receiveEvent); // register event
  Wire.onRequest(sendData);
  // set initial buffer values
    c[0] = mode;
    c[2] = r;
    c[3] = g;
    c[4] = b;
    c[5] = waitTime;
    c[6] = flag;
    c[7] = brightness;
}

void loop() {
  strip.setBrightness(brightness);
  switch(mode){
    case 0: 
      loadingOrange(waitTime, flag);
      break;
    case 1:
      loadingPurple(waitTime, flag);
      break;
    case 2: 
      volume(flag, waitTime, r, g, b, 0);
      break;
    case 3: 
      volume(flag, waitTime, r, g, b, 1);
      break;
    case 4:
      colorWipe(strip.Color(0, 0, 0), waitTime);
      colorWipe(strip.Color(r, g, b), waitTime);
      break;
    case 5:
      rainbow(waitTime);
      break;
    case 6:
      rainbowCycle(waitTime);
      break;
    case 7:
      error();
      break;
    case 8:
      warning();
      break;
    case 9:
      blinkRing(waitTime, r, g, b);
      break;
    default:
      standby();
  }
}
void error(){
      colorWipe(strip.Color(0, 0, 0), 15);
      colorWipe(strip.Color(255, 0, 0), 15);
}
void warning(){
  for (int i = 0; i < 16; i++){
    strip.setPixelColor(i, 255, 200, 0);
  }
  strip.show();
      colorWipe(strip.Color(0, 0, 0), 15);
      colorWipe(strip.Color(255, 255, 0), 15);
}
void blinkRing( uint8_t wait, uint8_t rc, uint8_t gc, uint8_t bc){
  for (int i = 0; i < 16; i++){
    strip.setPixelColor(i, rc, gc, bc);
  }
  strip.show();
  delay(wait);
  for (int i = 0; i < 16; i++){
    strip.setPixelColor(i, 0, 0, 0);
  }
  strip.show();
  delay(wait);
}
void standby(){
  // orange rgb(255,165,0)
  int R, G, B;
  for (int j = 1; j < 255; j++){
    R = map(j, 0, 255, 25, 254);
    G = map(j, 0, 255, 9, 70);
    B = map(j, 0, 255, 0, 0);
    for (int i = 0; i < 16; i ++){
      strip.setPixelColor(i, R, G, B);
    }
    strip.show();
    delay(6);
  }
    for (int j = 254; j > 0; j--){
    R = map(j, 0, 255, 25, 254);
    G = map(j, 0, 255, 9, 70);
    B = map(j, 0, 255, 0, 0);
    for (int i = 0; i < 16; i ++){
      strip.setPixelColor(i, R, G, B);
    }
    strip.show();
    delay(6);
  }
}
void volume(float p, uint8_t wait, uint8_t rc, uint8_t gc, uint8_t bc, bool d){
  bool ff;
  //calculate fade rate
  
  for (int i = 0; i <= 8-d; i ++){
    // see if led should be colored
    ff = ((float)i/9.0) < p/100.0;
    strip.setPixelColor(i, ff?rc:0, ff?gc:0, ff?bc:0);
    if(i < 8)
    strip.setPixelColor(d?8+i:16-i, ff?rc:0, ff?gc:0, ff?bc:0);
  }
  strip.show();
  delay(wait);
}
// Fill the dots one after the other with a color
void colorWipe(uint32_t c, uint8_t wait) {
  for(uint16_t i=0; i<strip.numPixels(); i++) {
      strip.setPixelColor(i, c);
      strip.show();
      delay(wait);
  }
}
 
void rainbow(uint8_t wait) {
  uint16_t i, j;
 
  for(j=0; j<256; j++) {
    for(i=0; i<strip.numPixels(); i++) {
      strip.setPixelColor(i, Wheel((i+j) & 255));
    }
    strip.show();
    delay(wait);
  }
}
// Slightly different, this makes the rainbow equally distributed throughout
void rainbowCycle(uint8_t wait) {
  uint16_t i, j;
 
  for(j=0; j<256*5; j++) { // 5 cycles of all colors on wheel
    for(i=0; i< strip.numPixels(); i++) {
      strip.setPixelColor(i, Wheel(((i * 256 / strip.numPixels()) + j) & 255));
    }
    strip.show();
    delay(wait);
  }
}
// Loading orange
void loadingOrange(uint8_t wait, uint8_t dir) {
  uint16_t i, j;
  float weightg[16] = {1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 0.9, 0.6, 0.3, 0.1};
  float weightb[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.7, 0.4, 0.1, 0.0, 0.0, 0};
  for(j=0; j<16; j++) {
    for(i=0; i < 16; i++){
      strip.setPixelColor((dir?(i+j):(i-j))%16, 
                           15.9*(dir?i:15-i), 
                           weightg[(dir?i:15-i)]*9*(dir?i:15-i),
                           weightb[(dir?i:15-i)]*15.9*(dir?i:15-i));
    }
    strip.show();
    delay(wait);
  }
}
// Loading purple
void loadingPurple(uint8_t wait, uint8_t dir) {
  uint16_t i, j;
  float weight[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.7, 0.5, 0.3, 0.2, 0};
  for(j=0; j<16; j++) {
    for(i=0; i < 16; i++){
      strip.setPixelColor((dir?(i+j):(i-j))%16, 
                           15.9*(dir?i:15-i), 
                           weight[(dir?i:15-i)]*15.9*(dir?i:15-i),
                           15.9*(dir?i:15-i));
    }
    strip.show();
    delay(wait);
  }
}
// Input a value 0 to 255 to get a color value.
// The colours are a transition r - g - b - back to r.
uint32_t Wheel(byte WheelPos) {
  if(WheelPos < 85) {
   return strip.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
  } else if(WheelPos < 170) {
   WheelPos -= 85;
   return strip.Color(255 - WheelPos * 3, 0, WheelPos * 3);
  } else {
   WheelPos -= 170;
   return strip.Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
}

void receiveEvent(int howMany) {
  while (0 < Wire.available()) { // loop through all but the last
    Wire.readBytes(c, howMany);
    mode = c[0];
    r = c[2];
    g = c[3];
    b = c[4];
    waitTime = c[5];
    flag = c[6];
    brightness = c[7];
  }
}

void sendData(){
  Wire.write(mode);
  Wire.write(r);
  Wire.write(g);
  Wire.write(b);
  Wire.write(waitTime);
  Wire.write(flag);
}
