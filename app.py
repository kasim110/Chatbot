#from crypt import methods
#chatbox-icon.svg , fill="#581B98"
from urllib import response
from flask import Flask, render_template,request,jsonify
from matplotlib.pyplot import text

from chat import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    open("msg.txt","w").close()
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    open('msg.txt','w').close()
    app.run(debug=True)
    

