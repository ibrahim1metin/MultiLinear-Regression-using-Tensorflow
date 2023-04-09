import tensorflow as tf
import numpy as np
x1=np.float32(np.random.normal(size=(1000,1),loc=3,scale=5))
x2=np.float32(np.random.normal(size=(1000,1),loc=-8,scale=10))
x3=np.float32(np.random.normal(size=(1000,1)))
x=[x1,x2,x3]
x=tf.constant(x)
y=0.613694*x1+2.27841*x2+1.2224232*x3+9
y=tf.constant(y)
x=tf.transpose(tf.squeeze(x))
model=tf.keras.Sequential([tf.keras.Input((None,3)),
                           tf.keras.layers.Dense(1)
])
model.compile(metrics=tf.keras.metrics.MeanSquaredError(),loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
model.fit(x,y,batch_size=100,epochs=800)
print(x[234],x1[234],x2[234],x3[234])
print(model.predict(tf.constant([[1],[2],[3]],tf.float32,(1,3)),batch_size=1))
print(0.613694*1+2.27841*2+1.2224232*3+9)
print(model.weights)
#Home-made LinReg
class LinReg:
    def __init__(self,datax:tf.constant,datay:tf.constant):
        self.datax=datax
        self.datay=datay
    def train(self,epochs,batch_size,learning_rate):
        self.initialize()
        epochSize=self.datax.shape[0]//batch_size
        for epoch in range(epochs):
            epochLoss=[]
            for step in range(epochSize):
                xval=tf.Variable(self.datax[batch_size*step:(step+1)*batch_size:],dtype=tf.float32)
                yval=tf.Variable(self.datay[batch_size*step:(step+1)*batch_size:],dtype=tf.float32)
                biasInUse=tf.Variable([self.b for _ in range(batch_size)])
                vals=[self.w,biasInUse]
                with tf.GradientTape() as tape:
                    tape.watch(vals)
                    pred=tf.matmul(xval,self.w)+biasInUse
                    lossVal=tf.constant(self.lossFunc(yval,pred))
                    epochLoss.append(lossVal)
                gradW,gradB=tape.gradient(lossVal,vals)
                self.w=self.w-gradW*learning_rate
                self.b=self.b-gradB[0]*learning_rate
                self.trainables=[self.w,self.b]
            meanLoss=tf.reduce_mean(epochLoss)
            print(f'{epoch+1}:{meanLoss}')
    def initialize(self):
        self.w=tf.Variable(tf.random.normal((self.datax.shape[-1],self.datay.shape[-1]),dtype=tf.float32))
        self.b=tf.Variable(1.)
        self.lossFunc=lambda yt,yp: tf.reduce_mean(tf.square(yt-yp))
        self.trainables=[self.w,self.b]
    def getTrainables(self):
        return self.trainables
    def predict(self,x):
        return tf.matmul(x,self.w)+self.b
lr=LinReg(x,y)
lr.train(200,10,0.005)
print(lr.getTrainables())
print(lr.predict(tf.reshape(tf.constant(x[2]),(1,3))),y[2])
