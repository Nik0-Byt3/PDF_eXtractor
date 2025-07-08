from utils.app import App
from flask import render_template, request, jsonify, Response, stream_with_context
import time 



# Global list to simulate log messages
log_queue = []

def log_to_client(msg):
    print(msg)  # still prints to terminal
    log_queue.append(msg)
    
def clear_log():
    global log_queue
    log_queue = []
    return jsonify({"status": "Log cleared"}), 200

@App.route('/stream')
def stream():
    def event_stream():
        last_sent = 0
        while True:
            if len(log_queue) > last_sent:
                new_msg = log_queue[last_sent]
                yield f"data: {new_msg}\n\n"
                last_sent += 1
            time.sleep(0.5)  # throttle for stability

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')




