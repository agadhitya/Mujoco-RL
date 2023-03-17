//sudo chmod a+rw /dev/ttyACM0

#define in1 3
#define in2 4
#define enA 5

#define in3 7
#define in4 8
#define enB 6

int start;
void setup() {
  // put your setup code here, to run once:
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(enA, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(enB, OUTPUT);

  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void loop() {
  // put your main code here, to run repeatedly:
//  for(int i = 150; i<=255; i=1+10){
      analogWrite(enA,200);
      analogWrite(enB,100);
//    delay(1000);
//  }

}
