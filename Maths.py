import math
import time
import random as rn

def Abs(v):
    return([abs(x) for x in v])

def Acos(x):
    return(math.acos(min(1,max(-1,x))))

def Add(v1,v2):
    return([v1[i]+v2[i] for i in range(len(v1))])

def Angle(v1,v2):
    return(Acos(Dot(v1,v2)/(Mag(v1)*Mag(v2))))

def AngleP(p1,p2,p3):
    return(Angle(Sub(p1,p2),Sub(p3,p2)))

def Asin(x):
    return(math.asin(min(1,max(-1,x))))

def Atan(x):
    return(math.atan(x))

def Avg(L):
    return(Scale([sum([L[u][i] for u in range(len(L))]) for i in range(len(L[0]))],1/len(L)))

def AvgC(L):
    return(sum(L)/len(L))

def Base(x, b):
    s = ''
    while x >= b:
        if len(str(x%b)) > 1:
            s += chr(x%b+55)
        else:
            s += str(x%b)
        x = x//b
    if len(str(x)) > 1:
        s += chr(x+55)
    else:
        s += str(x)
    return(s[::-1])

def Choose(n,r):
    p = 1
    for i in range(1,min(n-r+1,r+1)):
        p *= (n+1-i)/i
    return(int(p))

def CollatzCount(x):
    c = 1
    while x != 1:
        if x%2 == 0:
            x /= 2
        else:
            x = x*3 + 1
        c += 1
    return(c)

def Coprime(a,b):
    return([x for x in PrimeFactors(a) if x in PrimeFactors(b)] == [])

def Cos(x):
    return(math.cos(x))

def Cross(v1,v2):
    return([v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]])

def Decimal(t):
    if type(t) == int:
        return(t)
    elif t[1] == 1:
        return(t[0])
    else:
        return(t[0]/t[1])

def Dist(p1,p2):
    return((sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))**0.5)

def Dot(v1,v2):
    return(sum([v1[i]*v2[i] for i in range(len(v1))]))

def Factorial(x):
    if x == 0:
        return(1)
    p = x
    for i in range(1,x):
        p *= i
    return(p)

def Factors(x):
    facs = []
    for i in range(1,int(x**0.5)+1):
        if x%i == 0:
            facs.append(i)
            if i*i != x:
                facs.append(int(x/i))
    facs.sort()
    return(facs)

def Fibonacci(x):
    a = 0
    b = 1
    for i in range(x-1):
        a,b = b,a+b
    return(b)

def Fraction(x):
    i = 1
    while True:
        if abs(x*i - int(x*i)) < 10**(-10):
            return((int(x*i), i))
        if i > 1000:
            return(x)
        i += 1

def GCD(x,y):
    if y == 0:
        return(x)
    else:
        return(GCD(y,x%y))

def Int(v):
    return([int(i) for i in v])

def IsLeap(Y):
    return((Y%4 == 0 and Y%100 != 0) or Y%400 == 0)

def IsPrime(x):
    if x < 2 or x%2 == 0:
        return(False)
    for i in range(3,int(x**0.5)+1,2):
        if x%i == 0:
            return(False)
    return(True)

def LCM(L):
    if len(L) == 2:
        x = Fraction(L[0])
        y = Fraction(L[1])
        n = (int(abs(x[0]*y[0])/GCD(x[0],y[0])) , GCD(x[1], y[1]))
        if n[1] == 1:
            return(n[0])
        else:
            return(n)
    else:
        s = LCM(L[:2])
        if type(s) != int:
            s = s[0]/s[1]
        L = L[2:]
        L.insert(0,s)
        return(LCM(L))

def Lerp(A,B,t):
    if type(A) in [float,int]:
        return(A*(1-t)+B*t)
    return([A[i]*(1-t)+B[i]*t for i in range(len(A))])

def Log(x,b):
    return(math.log(x,b))

def Mag(v):
    return((sum([(i**2) for i in v])**.5))

def MDet(M):
    if len(M) == 2:
        return(M[0][0]*M[1][1] - M[0][1]*M[1][0])
    sm = 0
    for c in range(len(M)):
        nM = []
        for i in range(len(M)):
            if i != c:
                nM.append(M[i][1:])
        sm += (2*(1-c%2)-1)*M[c][0]*MDet(nM)
    return(sm)

def MInverse(M):
    s = len(M)
    I = [[int(r == c) for r in range(s)] for c in range(s)]
    while True in [(0 in m) for m in M]:
        a,b = rn.randint(0,s-1),rn.randint(0,s-1)
        while a == b:
            b = rn.randint(0,s-1)
        c = 0
        while c == 0:
            c = rn.randint(-5,5)
        for i in range(s):
            M[i][b] += c*M[i][a]
            I[i][b] += c*I[i][a]
    for r in range(s):
        for i in range(r,s):
            k = M[r][i]
            for u in range(s):
                M[u][i] /= k
                I[u][i] /= k
        for i in range(r+1,s):
            for u in range(s):
                M[u][i] -= M[u][r]
                I[u][i] -= I[u][r]
    for r in range(s-1,0,-1):
        for i in range(r):
            k = M[r][i]
            for u in range(s):
                M[u][i] -= k*M[u][r]
                I[u][i] -= k*I[u][r]
    return(I)

def MMmult(M1,M2):
    return([[sum([M1[i][r]*M2[c][i] for i in range(len(M1))]) for r in range(len(M1[0]))] for c in range(len(M2))])

def Mod(v,x):
    return([a%x for a in v])

def Mode(L):
    md = 0
    maxc = 0
    for i in L:
        cnt = L.count(i)
        if cnt > maxc:
            md = i
            maxc = cnt
    return((md,maxc))

def ModP(a,b,n):
    if n == 1:
        return(0)
    else:
        r = 1
        a = a%n
        while b > 0:
            if b%2 == 1:
                r = (r*a)%n
            b = b//2
            a = (a**2)%n
        return(r)

def MTranspose(M):
    return([[M[j][i] for j in range(len(M[0]))] for i in range(len(M))])

def MVmult(M,v):
    return([sum([M[i][r]*v[i] for i in range(len(M))]) for r in range(len(M[0]))])

def NextPrime(x):
    if x == 2:
        return(3)
    while True:
        x += 2
        if IsPrime(x):
            return(x)

def Normalized(v):
    if Mag(v) == 0:
        return(v)
    return(Scale(v,1/Mag(v)))

def OrthNorm(L):
    W = [L[0]]
    for i in range(1,len(L)):
        W.append(Sub(L[i], Sum([Proj(L[i],W[u]) for u in range(i)])))
    return([Normalized(w) for w in W])

def PassByV(p0,p1,an):
    pa = Sub(an,p0)
    pn = Sub(p1,p0)
    return(Sub(Proj(pa,pn),pa))
    

def Permutations(L):
    import itertools
    P = list(itertools.permutations(L))
    P.sort()
    return(P)

def PermutationsD(x):
    s = str(x)
    L = [s[i] for i in range(len(s))]
    P = Permutations(L)
    R = []
    for p in P:
        st = ''
        for i in p:
            st = st+i
        R.append(int(st))
    return(R)
            
def PrimeFactors(x):
    facs = []
    p = 2
    while x != 1:
        if x%p == 0:
            facs.append(p)
            x = int(x/p)
        else:
            p = NextPrime(p)
        if IsPrime(x):
           return(facs+[x])
    return(facs)

def Proj(a,b):
    return(Scale(b, Dot(a,b)/(Mag(b)**2)))

def ProjS(a,b):
    return(Dot(a,b)/Mag(b))

def Scale(v,c):
    return([v[i]*c for i in range(len(v))])

def Sign(x):
    return(int(x >= 0)*2 - 1)

def Sin(x):
    return(math.sin(x))

def Sub(v1,v2):
    return([v1[i]-v2[i] for i in range(len(v1))])

def Sum(V):
    return([sum(v[i] for v in V) for i in range(len(V[0]))])

def Tan(x):
    return(math.tan(x))

def Time(f,n,mxt,itr):
    x = 2
    while True:
        t = time.time()
        a = f(x)
        if time.time()-t > mxt:
            break
        x *= 2
    X,Y = [],[]
    for i in range(int(x/itr)*5,x,int(x/itr)):
        X.append(math.log(i))
        t = time.time()
        a = f(i)
        Y.append(math.log(max(time.time()-t,0.00001)))
    aX = sum(X)/len(X)
    aY = sum(Y)/len(Y)
    m = round(sum([(X[i]-aX)*(Y[i]-aY) for i in range(len(X))])/sum([(X[i]-aX)**2 for i in range(len(X))]))
    b = aY - m*aX
    return((math.e)**(m*(math.log(n)) + b))
    

def Totient(x):
    L = PrimeFactors(x)
    i = 0
    while i < len(L):
        while L.count(L[i]) > 1:
            L.remove(L[i])
        i += 1
    B = [Base(i,2) for i in range(2**len(L))]
    for i in range(len(B)):
        while len(B[i]) < len(L):
            B[i] = '0'+B[i]
    count = 0
    for b in B:
        k = 1
        for i in range(len(b)):
            k *= L[i]**int(b[i])
        count += (int(b.count('1')%2 == 0)*2 - 1) * int((x/k))
    return(abs(count))

def WeekDay(m,d,y):
    months = [31,28+int(IsLeap(y)),31,30,31,30,31,31,30,31,30,31]
    lys = 0
    dif = y-2017
    if y != 2017:
        sig = int(abs(dif)/dif)
    else:
        sig = 1
    for i in range(2017,y+sig,sig):
        lys += int(IsLeap(i))
    wd = (dif*365 + lys*sig + sum(months[:m-1]) + d -1)%7
    return(wd)