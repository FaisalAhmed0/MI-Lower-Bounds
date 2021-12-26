# MI-Lower-Bounds
This repo. contains a Pytorch implementation of four MI lower bounds based on the paper https://arxiv.org/pdf/1905.06922.pdf and the notebook https://github.com/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb \\

The definition of mutual information is </br>
<img src="https://latex.codecogs.com/svg.image?I(X;Y)&space;=&space;E_{p(x,y)}\Big[log&space;\frac{p(x,y)}{p(x)p(y)}\Big]" title="I(X;Y) = E_{p(x,y)}\Big[log \frac{p(x,y)}{p(x)p(y)}\Big]" /> </br>
The four MI lower bounds implemented are <br/>
<img src="https://latex.codecogs.com/svg.image?1.&space;I_{TUBA}&space;=&space;E_{p(x,y)}\Big[log&space;\frac{e^{f(x,y)}}{a(y)}\Big]&space;-&space;E_{p(x)p(y)}\Big[&space;\frac{e^{f(x,y)}}{a(y)}&space;\Big]" title="1. I_{TUBA} = E_{p(x,y)}\Big[log \frac{e^{f(x,y)}}{a(y)}\Big] - E_{p(x)p(y)}\Big[ \frac{e^{f(x,y)}}{a(y)} \Big]" /> <br/>

<img src="https://latex.codecogs.com/svg.image?2.I_{NWJ}&space;=&space;E_{p(x,y)}[f(x,y)]&space;-\frac{1}{e}&space;E_{p(x)p(y)}\Big[&space;e^{f(x,y)}&space;\Big]" title="2.I_{NWJ} = E_{p(x,y)}[f(x,y)] -\frac{1}{e} E_{p(x)p(y)}\Big[ e^{f(x,y)} \Big]" /> <br/>

<img src="https://latex.codecogs.com/svg.image?3.I_{NCE}&space;=&space;E_{p^K(x,y)}\Big[\frac{1}{K}&space;\sum_{i=1}^{K}log&space;\frac{e^{f(y_i,&space;x_i)}}{\frac{1}{K}\sum_{j=1}^{K}e^{f(y_i,x_j)}}\Big]" title="3.I_{NCE} = E_{p^K(x,y)}\Big[\frac{1}{K} \sum_{i=1}^{K}log \frac{e^{f(y_i, x_i)}}{\frac{1}{K}\sum_{j=1}^{K}e^{f(y_i,x_j)}}\Big]" />

<img src="https://latex.codecogs.com/svg.image?4.I_{\alpha}&space;=1&space;&plus;&space;\frac{1}{K}&space;\sum_{j=1}^{K}&space;\Big(&space;\frac{1}{K-1}&space;\sum_{i&space;\neq&space;j}\Big(&space;log\frac{e^{f(x_i,y_i)}}{a(y_i;x_{\neq&space;j})}&space;-&space;\frac{e^{f(x_i,y_j)}}{a(y_i;x_{\neq&space;j})}&space;\Big)&space;&space;\Big)" title="4.I_{\alpha} =1 + \frac{1}{K} \sum_{j=1}^{K} \Big( \frac{1}{K-1} \sum_{i \neq j}\Big( log\frac{e^{f(x_i,y_i)}}{a(y_i;x_{\neq j})} - \frac{e^{f(x_i,y_j)}}{a(y_i;x_{\neq j})} \Big) \Big)" /> <br/>
where <br/>
<img src="https://latex.codecogs.com/svg.image?a(y;&space;x_{1:M})&space;=&space;\alpha&space;\frac{1}{M}&space;\sum_{l=1}^{M}&space;e^{f(x_l,&space;y)}&space;&plus;&space;(1-\alpha)q(y)" title="a(y; x_{1:M}) = \alpha \frac{1}{M} \sum_{l=1}^{M} e^{f(x_l, y)} + (1-\alpha)q(y)" />




