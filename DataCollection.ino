void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);
//pinMode( 13, OUTPUT);
//pinMode( 11, OUTPUT);
//pinMode( 9, OUTPUT);
//pinMode( 7, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
int x=analogRead(A1);
Serial.println(x);
delay(1);

}
