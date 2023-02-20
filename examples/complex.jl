using Zygote
using FFTW

function compress(x, T)
    y = T * x
    println("toto")
    error()
    y
end

function forward(x, T)
    w = ifft(x)
    w = w .+ 1
    s = compress(w, T)
    y = fft(s)
    sum(abs2, y)
end

w = randn(10) .+ randn(10) .* im
T = randn(10, 10)

forward(w, T)

gradient(T -> forward(w, T), T)
