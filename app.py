from flask import Flask,request,render_template,url_for
import joblib
import sklearn

app=Flask(__name__)

model=joblib.load(r"D:\Data Sets\ML Models\spamq.joblib")
CV=joblib.load(r"D:\Data Sets\ML Models\spamcv.joblib")
#b=[]
@app.route("/")
def f():
    return render_template("spam.html")
@app.route("/result",methods=["GET","POST"])
def result():
    b=[]
    var1=list(str(request.form["text"]))
    b.append(''.join(var1))
    predict=model.predict(CV.transform(b))
    
    #if predict==0:
    #    print('ham')
    #else:
    #    print('Spam')
    return render_template('output.html',predict=predict)

if __name__=="__main__":
    app.run(debug=True)
