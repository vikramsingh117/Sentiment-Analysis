import paho.mqtt.client as mqtt
import random
import time

# MQTT Broker settings
broker = "broker.hivemq.com"
port = 1883
topic = "home/temperature"

# Publisher function for temperature readings
def publish_temperature():
    # Specify the callback API version explicitly
    client = mqtt.Client(client_id="Temperature_Sensor", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    client.connect(broker, port)
    client.loop_start()

    while True:
        temperature = round(random.uniform(20.0, 30.0), 2)  # Simulate temperature between 20.0 and 30.0°C
        message = f"{temperature}°C"
        result = client.publish(topic, message)
        status = result[0]

        if status == 0:
            print(f"Sent temperature `{message}` to topic `{topic}`")
        else:
            print(f"Failed to send temperature to topic `{topic}`")

        time.sleep(5)  # Publish every 5 seconds

    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    publish_temperature()
