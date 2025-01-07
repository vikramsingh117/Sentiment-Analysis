import paho.mqtt.client as mqtt

# MQTT Broker settings
broker = "broker.hivemq.com"
port = 1883
topic = "home/temperature"

# Callback when connected to broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}\n")

# Callback for receiving messages
def on_message(client, userdata, msg):
    print(f"Temperature reading received: {msg.payload.decode()}")

# Subscriber function
def subscribe_temperature():
    # Specify CallbackAPIVersion.VERSION1 to use the old callback style
    client = mqtt.Client(client_id="Temperature_Monitor", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, port)
    client.loop_forever()  # Keep the subscriber running

if __name__ == "__main__":
    subscribe_temperature()
