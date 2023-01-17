a = '123'
b = '123.456'

from execdata import convint as s2int, convfloat as s2dec

result_a = s2int(a)
print(result_a)
print(type(result_a))
result_b = s2dec(b)
print(result_b)
print(type(result_b))