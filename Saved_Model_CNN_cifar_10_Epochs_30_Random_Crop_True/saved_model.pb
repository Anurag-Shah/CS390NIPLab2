бу
«Ѓ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
њ
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
└
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.12v2.4.0-49-g85c8b2a817f8█Ж
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
ѓ
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
І
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(**
shared_namebatch_normalization/gamma
ё
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:└(*
dtype0
Ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(*)
shared_namebatch_normalization/beta
ѓ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:└(*
dtype0
Ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(*0
shared_name!batch_normalization/moving_mean
љ
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:└(*
dtype0
Ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(*4
shared_name%#batch_normalization/moving_variance
ў
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:└(*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└(ђ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
└(ђ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:ђ*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
ђђ*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:ђ*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	ђ
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
ї
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m
Ё
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
љ
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/m
Ѕ
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
ђ
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
Ў
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(*1
shared_name" Adam/batch_normalization/gamma/m
њ
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:└(*
dtype0
Ќ
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(*0
shared_name!Adam/batch_normalization/beta/m
љ
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:└(*
dtype0
ё
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└(ђ*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
└(ђ*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:ђ*
dtype0
ѕ
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*&
shared_nameAdam/dense_1/kernel/m
Ђ
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
ђђ*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*&
shared_nameAdam/dense_2/kernel/m
ђ
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	ђ
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:
*
dtype0
ї
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v
Ё
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
љ
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/v
Ѕ
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
ђ
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
Ў
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(*1
shared_name" Adam/batch_normalization/gamma/v
њ
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:└(*
dtype0
Ќ
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:└(*0
shared_name!Adam/batch_normalization/beta/v
љ
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:└(*
dtype0
ё
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└(ђ*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
└(ђ*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:ђ*
dtype0
ѕ
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*&
shared_nameAdam/dense_1/kernel/v
Ђ
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
ђђ*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*&
shared_nameAdam/dense_2/kernel/v
ђ
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	ђ
*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
СN
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЪN
valueЋNBњN BІN
Ф
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures

_rng
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
R
!trainable_variables
"regularization_losses
#	variables
$	keras_api
R
%trainable_variables
&regularization_losses
'	variables
(	keras_api
R
)trainable_variables
*regularization_losses
+	variables
,	keras_api
Ќ
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2trainable_variables
3regularization_losses
4	variables
5	keras_api
h

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
R
<trainable_variables
=regularization_losses
>	variables
?	keras_api
h

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
R
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
Ю
Piter

Qbeta_1

Rbeta_2
	SdecaymюmЮmъmЪ.mа/mА6mб7mБ@mцAmЦJmдKmДvеvЕvфvФ.vг/vГ6v«7v»@v░Av▒Jv▓Kv│
V
0
1
2
3
.4
/5
66
77
@8
A9
J10
K11
 
f
0
1
2
3
.4
/5
06
17
68
79
@10
A11
J12
K13
Г
trainable_variables
Tlayer_metrics
Ulayer_regularization_losses
regularization_losses
Vnon_trainable_variables
Wmetrics
	variables

Xlayers
 

Y
_state_var
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
trainable_variables
Zlayer_metrics
[layer_regularization_losses
regularization_losses
\non_trainable_variables
]metrics
	variables

^layers
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
trainable_variables
_layer_metrics
`layer_regularization_losses
regularization_losses
anon_trainable_variables
bmetrics
	variables

clayers
 
 
 
Г
!trainable_variables
dlayer_metrics
elayer_regularization_losses
"regularization_losses
fnon_trainable_variables
gmetrics
#	variables

hlayers
 
 
 
Г
%trainable_variables
ilayer_metrics
jlayer_regularization_losses
&regularization_losses
knon_trainable_variables
lmetrics
'	variables

mlayers
 
 
 
Г
)trainable_variables
nlayer_metrics
olayer_regularization_losses
*regularization_losses
pnon_trainable_variables
qmetrics
+	variables

rlayers
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
02
13
Г
2trainable_variables
slayer_metrics
tlayer_regularization_losses
3regularization_losses
unon_trainable_variables
vmetrics
4	variables

wlayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
Г
8trainable_variables
xlayer_metrics
ylayer_regularization_losses
9regularization_losses
znon_trainable_variables
{metrics
:	variables

|layers
 
 
 
»
<trainable_variables
}layer_metrics
~layer_regularization_losses
=regularization_losses
non_trainable_variables
ђmetrics
>	variables
Ђlayers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
▓
Btrainable_variables
ѓlayer_metrics
 Ѓlayer_regularization_losses
Cregularization_losses
ёnon_trainable_variables
Ёmetrics
D	variables
єlayers
 
 
 
▓
Ftrainable_variables
Єlayer_metrics
 ѕlayer_regularization_losses
Gregularization_losses
Ѕnon_trainable_variables
іmetrics
H	variables
Іlayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
▓
Ltrainable_variables
їlayer_metrics
 Їlayer_regularization_losses
Mregularization_losses
јnon_trainable_variables
Јmetrics
N	variables
љlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
 
 

00
11

Љ0
њ1
V
0
1
2
3
4
5
6
7
	8

9
10
11
PN
VARIABLE_VALUEVariable2layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

00
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Њtotal

ћcount
Ћ	variables
ќ	keras_api
I

Ќtotal

ўcount
Ў
_fn_kwargs
џ	variables
Џ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Њ0
ћ1

Ћ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ќ0
ў1

џ	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ћ
!serving_default_random_crop_inputPlaceholder*/
_output_shapes
:           *
dtype0*$
shape:           
у
StatefulPartitionedCallStatefulPartitionedCall!serving_default_random_crop_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/betabatch_normalization/gammadense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference_signature_wrapper_26565
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ќ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOpVariable/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*<
Tin5
321		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *'
f"R 
__inference__traced_save_27347
т	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayVariabletotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__traced_restore_27498╬ъ
▀
|
'__inference_dense_1_layer_call_fn_27131

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_260672
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ыc
ч
E__inference_sequential_layer_call_and_return_conditional_losses_26257
random_crop_input
conv2d_26217
conv2d_26219
conv2d_1_26222
conv2d_1_26224
batch_normalization_26230
batch_normalization_26232
batch_normalization_26234
batch_normalization_26236
dense_26239
dense_26241
dense_1_26245
dense_1_26247
dense_2_26251
dense_2_26253
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallg
random_crop/ShapeShaperandom_crop_input*
T0*
_output_shapes
:2
random_crop/Shapeї
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
random_crop/strided_slice/stackљ
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_1љ
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_2ф
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_sliceљ
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice_1/stackћ
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_1ћ
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_2┤
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_slice_1љ
random_crop/truediv/CastCast"random_crop/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv/Castќ
random_crop/truediv/Cast_1Cast$random_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv/Cast_1ћ
random_crop/truedivRealDivrandom_crop/truediv/Cast:y:0random_crop/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truedivw
random_crop/Greater/yConst*
_output_shapes
: *
dtype0*
valueB 2      ­?2
random_crop/Greater/yЈ
random_crop/GreaterGreaterrandom_crop/truediv:z:0random_crop/Greater/y:output:0*
T0*
_output_shapes
: 2
random_crop/Greaterћ
random_crop/condStatelessIfrandom_crop/Greater:z:0"random_crop/strided_slice:output:0$random_crop/strided_slice_1:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 */
else_branch R
random_crop_cond_false_26159*
output_shapes
: *.
then_branchR
random_crop_cond_true_261582
random_crop/cond~
random_crop/cond/IdentityIdentityrandom_crop/cond:output:0*
T0*
_output_shapes
: 2
random_crop/cond/Identityю
random_crop/cond_1StatelessIfrandom_crop/Greater:z:0$random_crop/strided_slice_1:output:0"random_crop/strided_slice:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *1
else_branch"R 
random_crop_cond_1_false_26178*
output_shapes
: *0
then_branch!R
random_crop_cond_1_true_261772
random_crop/cond_1ё
random_crop/cond_1/IdentityIdentityrandom_crop/cond_1:output:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identityд
random_crop/stackPack"random_crop/cond/Identity:output:0$random_crop/cond_1/Identity:output:0*
N*
T0*
_output_shapes
:2
random_crop/stackь
!random_crop/resize/ResizeBilinearResizeBilinearrandom_crop_inputrandom_crop/stack:output:0*
T0*A
_output_shapes/
-:+                           *
half_pixel_centers(2#
!random_crop/resize/ResizeBilinearh
random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/sub/yі
random_crop/subSub"random_crop/cond/Identity:output:0random_crop/sub/y:output:0*
T0*
_output_shapes
: 2
random_crop/subl
random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/sub_1/yњ
random_crop/sub_1Sub$random_crop/cond_1/Identity:output:0random_crop/sub_1/y:output:0*
T0*
_output_shapes
: 2
random_crop/sub_1t
random_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/truediv_1/yЁ
random_crop/truediv_1/CastCastrandom_crop/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_1/Castќ
random_crop/truediv_1/Cast_1Cast random_crop/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_1/Cast_1ю
random_crop/truediv_1RealDivrandom_crop/truediv_1/Cast:y:0 random_crop/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truediv_1w
random_crop/CastCastrandom_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/Castt
random_crop/truediv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/truediv_2/yЄ
random_crop/truediv_2/CastCastrandom_crop/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_2/Castќ
random_crop/truediv_2/Cast_1Cast random_crop/truediv_2/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_2/Cast_1ю
random_crop/truediv_2RealDivrandom_crop/truediv_2/Cast:y:0 random_crop/truediv_2/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truediv_2{
random_crop/Cast_1Castrandom_crop/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/Cast_1p
random_crop/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
random_crop/stack_1/0p
random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
value	B : 2
random_crop/stack_1/3╬
random_crop/stack_1Packrandom_crop/stack_1/0:output:0random_crop/Cast:y:0random_crop/Cast_1:y:0random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:2
random_crop/stack_1Ѓ
random_crop/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"              2
random_crop/stack_2Т
random_crop/SliceSlice2random_crop/resize/ResizeBilinear:resized_images:0random_crop/stack_1:output:0random_crop/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         2
random_crop/Sliceд
conv2d/StatefulPartitionedCallStatefulPartitionedCallrandom_crop/Slice:output:0conv2d_26217conv2d_26219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_258762 
conv2d/StatefulPartitionedCallй
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_26222conv2d_1_26224*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_259032"
 conv2d_1/StatefulPartitionedCallљ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_256762
max_pooling2d/PartitionedCallч
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_259372
dropout/PartitionedCallЬ
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_259562
flatten/PartitionedCallа
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_26230batch_normalization_26232batch_normalization_26234batch_normalization_26236*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_258112-
+batch_normalization/StatefulPartitionedCall┤
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26239dense_26241*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_260102
dense/StatefulPartitionedCallЩ
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_260432
dropout_1/PartitionedCallг
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_26245dense_1_26247*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_260672!
dense_1/StatefulPartitionedCallЧ
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_261002
dropout_2/PartitionedCallФ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_26251dense_2_26253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_261242!
dense_2/StatefulPartitionedCallм
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:           ::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:b ^
/
_output_shapes
:           
+
_user_specified_namerandom_crop_input
т

▒
random_crop_cond_true_267192
.random_crop_cond_mul_random_crop_strided_slice?
;random_crop_cond_truediv_cast_1_random_crop_strided_slice_1
random_crop_cond_identityr
random_crop/cond/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond/mul/xЦ
random_crop/cond/mulMulrandom_crop/cond/mul/x:output:0.random_crop_cond_mul_random_crop_strided_slice*
T0*
_output_shapes
: 2
random_crop/cond/mulљ
random_crop/cond/truediv/CastCastrandom_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond/truediv/Castи
random_crop/cond/truediv/Cast_1Cast;random_crop_cond_truediv_cast_1_random_crop_strided_slice_1*

DstT0*

SrcT0*
_output_shapes
: 2!
random_crop/cond/truediv/Cast_1е
random_crop/cond/truedivRealDiv!random_crop/cond/truediv/Cast:y:0#random_crop/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/cond/truedivё
random_crop/cond/CastCastrandom_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond/Cast~
random_crop/cond/IdentityIdentityrandom_crop/cond/Cast:y:0*
T0*
_output_shapes
: 2
random_crop/cond/Identity"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
Ъ
║
random_crop_cond_1_false_261786
2random_crop_cond_1_mul_random_crop_strided_slice_1?
;random_crop_cond_1_truediv_cast_1_random_crop_strided_slice
random_crop_cond_1_identityv
random_crop/cond_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond_1/mul/x»
random_crop/cond_1/mulMul!random_crop/cond_1/mul/x:output:02random_crop_cond_1_mul_random_crop_strided_slice_1*
T0*
_output_shapes
: 2
random_crop/cond_1/mulќ
random_crop/cond_1/truediv/CastCastrandom_crop/cond_1/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2!
random_crop/cond_1/truediv/Cast╗
!random_crop/cond_1/truediv/Cast_1Cast;random_crop_cond_1_truediv_cast_1_random_crop_strided_slice*

DstT0*

SrcT0*
_output_shapes
: 2#
!random_crop/cond_1/truediv/Cast_1░
random_crop/cond_1/truedivRealDiv#random_crop/cond_1/truediv/Cast:y:0%random_crop/cond_1/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/cond_1/truedivі
random_crop/cond_1/CastCastrandom_crop/cond_1/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond_1/Castё
random_crop/cond_1/IdentityIdentityrandom_crop/cond_1/Cast:y:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identity"C
random_crop_cond_1_identity$random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
Ќ

к
*__inference_sequential_layer_call_fn_26871

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_263422
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:           :::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
ќ
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_26038

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedм2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
г
і
random_crop_cond_1_true_26413"
random_crop_cond_1_placeholder$
 random_crop_cond_1_placeholder_1
random_crop_cond_1_identityv
random_crop/cond_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond_1/Constі
random_crop/cond_1/IdentityIdentity!random_crop/cond_1/Const:output:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identity"C
random_crop_cond_1_identity$random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
╦

┌
A__inference_conv2d_layer_call_and_return_conditional_losses_26915

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ќ
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_27143

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedм2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Э	
Х
*__inference_sequential_layer_call_fn_26904

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_264932
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:           ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
Ц
b
)__inference_dropout_1_layer_call_fn_27106

inputs
identityѕбStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_260382
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
з	
┘
@__inference_dense_layer_call_and_return_conditional_losses_27075

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└(ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
Ќ
Ѓ
random_crop_cond_false_26395 
random_crop_cond_placeholder"
random_crop_cond_placeholder_1
random_crop_cond_identityr
random_crop/cond/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond/Constё
random_crop/cond/IdentityIdentityrandom_crop/cond/Const:output:0*
T0*
_output_shapes
: 2
random_crop/cond/Identity"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
г
і
random_crop_cond_1_true_26738"
random_crop_cond_1_placeholder$
 random_crop_cond_1_placeholder_1
random_crop_cond_1_identityv
random_crop/cond_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond_1/Constі
random_crop/cond_1/IdentityIdentity!random_crop/cond_1/Const:output:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identity"C
random_crop_cond_1_identity$random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
П
|
'__inference_dense_2_layer_call_fn_27178

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_261242
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
▒
C
'__inference_dropout_layer_call_fn_26971

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_259372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         		@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
И
д
3__inference_batch_normalization_layer_call_fn_27064

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_258112
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └(::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
Љ
»
'sequential_random_crop_cond_false_25554+
'sequential_random_crop_cond_placeholder-
)sequential_random_crop_cond_placeholder_1(
$sequential_random_crop_cond_identityѕ
!sequential/random_crop/cond/ConstConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/random_crop/cond/ConstЦ
$sequential/random_crop/cond/IdentityIdentity*sequential/random_crop/cond/Const:output:0*
T0*
_output_shapes
: 2&
$sequential/random_crop/cond/Identity"U
$sequential_random_crop_cond_identity-sequential/random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
т

▒
random_crop_cond_true_261582
.random_crop_cond_mul_random_crop_strided_slice?
;random_crop_cond_truediv_cast_1_random_crop_strided_slice_1
random_crop_cond_identityr
random_crop/cond/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond/mul/xЦ
random_crop/cond/mulMulrandom_crop/cond/mul/x:output:0.random_crop_cond_mul_random_crop_strided_slice*
T0*
_output_shapes
: 2
random_crop/cond/mulљ
random_crop/cond/truediv/CastCastrandom_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond/truediv/Castи
random_crop/cond/truediv/Cast_1Cast;random_crop_cond_truediv_cast_1_random_crop_strided_slice_1*

DstT0*

SrcT0*
_output_shapes
: 2!
random_crop/cond/truediv/Cast_1е
random_crop/cond/truedivRealDiv!random_crop/cond/truediv/Cast:y:0#random_crop/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/cond/truedivё
random_crop/cond/CastCastrandom_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond/Cast~
random_crop/cond/IdentityIdentityrandom_crop/cond/Cast:y:0*
T0*
_output_shapes
: 2
random_crop/cond/Identity"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
Ў
E
)__inference_dropout_2_layer_call_fn_27158

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_261002
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
щ
{
&__inference_conv2d_layer_call_fn_26924

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_258762
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Б
C
'__inference_flatten_layer_call_fn_26982

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_259562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
т
`
B__inference_dropout_layer_call_and_return_conditional_losses_25937

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         		@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         		@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
╦

┌
A__inference_conv2d_layer_call_and_return_conditional_losses_25876

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
а/
Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27018

inputs
assignmovingavg_26993
assignmovingavg_1_26999 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	└(*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	└(2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         └(2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	└(*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└(*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└(*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/26993*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26993*
_output_shapes	
:└(*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/26993*
_output_shapes	
:└(2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/26993*
_output_shapes	
:└(2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26993AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/26993*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/26999*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26999*
_output_shapes	
:└(*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/26999*
_output_shapes	
:└(2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/26999*
_output_shapes	
:└(2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26999AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/26999*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└(2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└(2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:└(2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└(2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└(2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └(::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
═

▄
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26935

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ђ
Ч
)sequential_random_crop_cond_1_false_25573L
Hsequential_random_crop_cond_1_mul_sequential_random_crop_strided_slice_1U
Qsequential_random_crop_cond_1_truediv_cast_1_sequential_random_crop_strided_slice*
&sequential_random_crop_cond_1_identityї
#sequential/random_crop/cond_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/random_crop/cond_1/mul/xТ
!sequential/random_crop/cond_1/mulMul,sequential/random_crop/cond_1/mul/x:output:0Hsequential_random_crop_cond_1_mul_sequential_random_crop_strided_slice_1*
T0*
_output_shapes
: 2#
!sequential/random_crop/cond_1/mulи
*sequential/random_crop/cond_1/truediv/CastCast%sequential/random_crop/cond_1/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*sequential/random_crop/cond_1/truediv/Castу
,sequential/random_crop/cond_1/truediv/Cast_1CastQsequential_random_crop_cond_1_truediv_cast_1_sequential_random_crop_strided_slice*

DstT0*

SrcT0*
_output_shapes
: 2.
,sequential/random_crop/cond_1/truediv/Cast_1▄
%sequential/random_crop/cond_1/truedivRealDiv.sequential/random_crop/cond_1/truediv/Cast:y:00sequential/random_crop/cond_1/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2'
%sequential/random_crop/cond_1/truedivФ
"sequential/random_crop/cond_1/CastCast)sequential/random_crop/cond_1/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"sequential/random_crop/cond_1/CastЦ
&sequential/random_crop/cond_1/IdentityIdentity&sequential/random_crop/cond_1/Cast:y:0*
T0*
_output_shapes
: 2(
&sequential/random_crop/cond_1/Identity"Y
&sequential_random_crop_cond_1_identity/sequential/random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
ь	
║
#__inference_signature_wrapper_26565
random_crop_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallrandom_crop_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__wrapped_model_256702
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:           ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:           
+
_user_specified_namerandom_crop_input
а/
Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25778

inputs
assignmovingavg_25753
assignmovingavg_1_25759 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	└(*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	└(2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         └(2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	└(*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└(*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└(*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/25753*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25753*
_output_shapes	
:└(*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/25753*
_output_shapes	
:└(2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/25753*
_output_shapes	
:└(2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25753AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/25753*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/25759*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25759*
_output_shapes	
:└(*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/25759*
_output_shapes	
:└(2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/25759*
_output_shapes	
:└(2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25759AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/25759*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└(2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└(2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:└(2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└(2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└(2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └(::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
т
`
B__inference_dropout_layer_call_and_return_conditional_losses_26961

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         		@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         		@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
═
a
B__inference_dropout_layer_call_and_return_conditional_losses_25932

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         		@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╔
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         		@*
dtype0*
seedм2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         		@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         		@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         		@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         		@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
╦
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_27148

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
э	
█
B__inference_dense_2_layer_call_and_return_conditional_losses_27169

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
═
a
B__inference_dropout_layer_call_and_return_conditional_losses_26956

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         		@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╔
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         		@*
dtype0*
seedм2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         		@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         		@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         		@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         		@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
э	
█
B__inference_dense_2_layer_call_and_return_conditional_losses_26124

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ъ
║
random_crop_cond_1_false_267396
2random_crop_cond_1_mul_random_crop_strided_slice_1?
;random_crop_cond_1_truediv_cast_1_random_crop_strided_slice
random_crop_cond_1_identityv
random_crop/cond_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond_1/mul/x»
random_crop/cond_1/mulMul!random_crop/cond_1/mul/x:output:02random_crop_cond_1_mul_random_crop_strided_slice_1*
T0*
_output_shapes
: 2
random_crop/cond_1/mulќ
random_crop/cond_1/truediv/CastCastrandom_crop/cond_1/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2!
random_crop/cond_1/truediv/Cast╗
!random_crop/cond_1/truediv/Cast_1Cast;random_crop_cond_1_truediv_cast_1_random_crop_strided_slice*

DstT0*

SrcT0*
_output_shapes
: 2#
!random_crop/cond_1/truediv/Cast_1░
random_crop/cond_1/truedivRealDiv#random_crop/cond_1/truediv/Cast:y:0%random_crop/cond_1/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/cond_1/truedivі
random_crop/cond_1/CastCastrandom_crop/cond_1/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond_1/Castё
random_crop/cond_1/IdentityIdentityrandom_crop/cond_1/Cast:y:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identity"C
random_crop_cond_1_identity$random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
╦
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_26043

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
█
z
%__inference_dense_layer_call_fn_27084

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_260102
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └(::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
ш	
█
B__inference_dense_1_layer_call_and_return_conditional_losses_26067

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╩l
П
E__inference_sequential_layer_call_and_return_conditional_losses_26141
random_crop_input2
.random_crop_stateful_uniform_full_int_resource
conv2d_25887
conv2d_25889
conv2d_1_25914
conv2d_1_25916
batch_normalization_25990
batch_normalization_25992
batch_normalization_25994
batch_normalization_25996
dense_26021
dense_26023
dense_1_26078
dense_1_26080
dense_2_26135
dense_2_26137
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallбrandom_crop/Assert/Assertб%random_crop/stateful_uniform_full_intg
random_crop/ShapeShaperandom_crop_input*
T0*
_output_shapes
:2
random_crop/Shapeї
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
random_crop/strided_slice/stackљ
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_1љ
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_2ф
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_sliceљ
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice_1/stackћ
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_1ћ
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_2┤
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_slice_1l
random_crop/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/stack/1l
random_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/stack/2Р
random_crop/stackPack"random_crop/strided_slice:output:0random_crop/stack/1:output:0random_crop/stack/2:output:0$random_crop/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_crop/stackА
random_crop/GreaterEqualGreaterEqualrandom_crop/Shape:output:0random_crop/stack:output:0*
T0*
_output_shapes
:2
random_crop/GreaterEqualp
random_crop/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
random_crop/Const{
random_crop/AllAllrandom_crop/GreaterEqual:z:0random_crop/Const:output:0*
_output_shapes
: 2
random_crop/Allv
random_crop/Assert/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/Assert/Constz
random_crop/Assert/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/Assert/Const_1є
 random_crop/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 random_crop/Assert/Assert/data_0є
 random_crop/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 random_crop/Assert/Assert/data_1М
random_crop/Assert/AssertAssertrandom_crop/All:output:0)random_crop/Assert/Assert/data_0:output:0)random_crop/Assert/Assert/data_1:output:0*
T
2*
_output_shapes
 2
random_crop/Assert/Assert¤
random_crop/control_dependencyIdentityrandom_crop/Shape:output:0^random_crop/Assert/Assert*
T0*$
_class
loc:@random_crop/Shape*
_output_shapes
:2 
random_crop/control_dependencyЊ
random_crop/subSub'random_crop/control_dependency:output:0random_crop/stack:output:0*
T0*
_output_shapes
:2
random_crop/subh
random_crop/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/add/yЂ
random_crop/addAddV2random_crop/sub:z:0random_crop/add/y:output:0*
T0*
_output_shapes
:2
random_crop/addt
random_crop/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
random_crop/Shape_1ц
+random_crop/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_crop/stateful_uniform_full_int/shapeц
/random_crop/stateful_uniform_full_int/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/random_crop/stateful_uniform_full_int/algorithm─
%random_crop/stateful_uniform_full_intStatefulUniformFullInt.random_crop_stateful_uniform_full_int_resource8random_crop/stateful_uniform_full_int/algorithm:output:04random_crop/stateful_uniform_full_int/shape:output:0*
_output_shapes
:*
dtype0	*
shape_dtype02'
%random_crop/stateful_uniform_full_intz
random_crop/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_crop/zeros_likeх
random_crop/stack_1Pack.random_crop/stateful_uniform_full_int:output:0random_crop/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_crop/stack_1Ќ
!random_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_crop/strided_slice_2/stackЏ
#random_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_crop/strided_slice_2/stack_1Џ
#random_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_crop/strided_slice_2/stack_2▄
random_crop/strided_slice_2StridedSlicerandom_crop/stack_1:output:0*random_crop/strided_slice_2/stack:output:0,random_crop/strided_slice_2/stack_1:output:0,random_crop/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_crop/strided_slice_2ќ
(random_crop/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2*
(random_crop/stateless_random_uniform/minџ
(random_crop/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :    2*
(random_crop/stateless_random_uniform/max┼
$random_crop/stateless_random_uniformStatelessRandomUniformIntrandom_crop/Shape_1:output:0$random_crop/strided_slice_2:output:01random_crop/stateless_random_uniform/min:output:01random_crop/stateless_random_uniform/max:output:0*
T0*
_output_shapes
:*
dtype02&
$random_crop/stateless_random_uniformЌ
random_crop/modFloorMod-random_crop/stateless_random_uniform:output:0random_crop/add:z:0*
T0*
_output_shapes
:2
random_crop/mod║
random_crop/SliceSlicerandom_crop_inputrandom_crop/mod:z:0random_crop/stack:output:0*
Index0*
T0*/
_output_shapes
:         2
random_crop/Sliceд
conv2d/StatefulPartitionedCallStatefulPartitionedCallrandom_crop/Slice:output:0conv2d_25887conv2d_25889*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_258762 
conv2d/StatefulPartitionedCallй
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_25914conv2d_1_25916*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_259032"
 conv2d_1/StatefulPartitionedCallљ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_256762
max_pooling2d/PartitionedCall»
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0^random_crop/Assert/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_259322!
dropout/StatefulPartitionedCallШ
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_259562
flatten/PartitionedCallъ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_25990batch_normalization_25992batch_normalization_25994batch_normalization_25996*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_257782-
+batch_normalization/StatefulPartitionedCall┤
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26021dense_26023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_260102
dense/StatefulPartitionedCall┤
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_260382#
!dropout_1/StatefulPartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_26078dense_1_26080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_260672!
dense_1/StatefulPartitionedCallИ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_260952#
!dropout_2/StatefulPartitionedCall│
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_26135dense_2_26137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_261242!
dense_2/StatefulPartitionedCallђ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^random_crop/Assert/Assert&^random_crop/stateful_uniform_full_int*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:           :::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall26
random_crop/Assert/Assertrandom_crop/Assert/Assert2N
%random_crop/stateful_uniform_full_int%random_crop/stateful_uniform_full_int:b ^
/
_output_shapes
:           
+
_user_specified_namerandom_crop_input
Ќ
Ѓ
random_crop_cond_false_26159 
random_crop_cond_placeholder"
random_crop_cond_placeholder_1
random_crop_cond_identityr
random_crop/cond/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond/Constё
random_crop/cond/IdentityIdentityrandom_crop/cond/Const:output:0*
T0*
_output_shapes
: 2
random_crop/cond/Identity"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
г
і
random_crop_cond_1_true_26177"
random_crop_cond_1_placeholder$
 random_crop_cond_1_placeholder_1
random_crop_cond_1_identityv
random_crop/cond_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond_1/Constі
random_crop/cond_1/IdentityIdentity!random_crop/cond_1/Const:output:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identity"C
random_crop_cond_1_identity$random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
ѕa
║
__inference__traced_save_27347
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop'
#savev2_variable_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameх
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*К
valueйB║0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesУ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЧ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop#savev2_variable_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220		2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ю
_input_shapesі
Є: : : : @:@:└(:└(:└(:└(:
└(ђ:ђ:
ђђ:ђ:	ђ
:
: : : : :: : : : : : : @:@:└(:└(:
└(ђ:ђ:
ђђ:ђ:	ђ
:
: : : @:@:└(:└(:
└(ђ:ђ:
ђђ:ђ:	ђ
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:!

_output_shapes	
:└(:!

_output_shapes	
:└(:!

_output_shapes	
:└(:!

_output_shapes	
:└(:&	"
 
_output_shapes
:
└(ђ:!


_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:!

_output_shapes	
:└(:!

_output_shapes	
:└(:&"
 
_output_shapes
:
└(ђ:!

_output_shapes	
:ђ:& "
 
_output_shapes
:
ђђ:!!

_output_shapes	
:ђ:%"!

_output_shapes
:	ђ
: #

_output_shapes
:
:,$(
&
_output_shapes
: : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@:!(

_output_shapes	
:└(:!)

_output_shapes	
:└(:&*"
 
_output_shapes
:
└(ђ:!+

_output_shapes	
:ђ:&,"
 
_output_shapes
:
ђђ:!-

_output_shapes	
:ђ:%.!

_output_shapes
:	ђ
: /

_output_shapes
:
:0

_output_shapes
: 
К
з
&sequential_random_crop_cond_true_25553H
Dsequential_random_crop_cond_mul_sequential_random_crop_strided_sliceU
Qsequential_random_crop_cond_truediv_cast_1_sequential_random_crop_strided_slice_1(
$sequential_random_crop_cond_identityѕ
!sequential/random_crop/cond/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/random_crop/cond/mul/x▄
sequential/random_crop/cond/mulMul*sequential/random_crop/cond/mul/x:output:0Dsequential_random_crop_cond_mul_sequential_random_crop_strided_slice*
T0*
_output_shapes
: 2!
sequential/random_crop/cond/mul▒
(sequential/random_crop/cond/truediv/CastCast#sequential/random_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(sequential/random_crop/cond/truediv/Castс
*sequential/random_crop/cond/truediv/Cast_1CastQsequential_random_crop_cond_truediv_cast_1_sequential_random_crop_strided_slice_1*

DstT0*

SrcT0*
_output_shapes
: 2,
*sequential/random_crop/cond/truediv/Cast_1н
#sequential/random_crop/cond/truedivRealDiv,sequential/random_crop/cond/truediv/Cast:y:0.sequential/random_crop/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2%
#sequential/random_crop/cond/truedivЦ
 sequential/random_crop/cond/CastCast'sequential/random_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 sequential/random_crop/cond/CastЪ
$sequential/random_crop/cond/IdentityIdentity$sequential/random_crop/cond/Cast:y:0*
T0*
_output_shapes
: 2&
$sequential/random_crop/cond/Identity"U
$sequential_random_crop_cond_identity-sequential/random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
д
Х
(sequential_random_crop_cond_1_true_25572-
)sequential_random_crop_cond_1_placeholder/
+sequential_random_crop_cond_1_placeholder_1*
&sequential_random_crop_cond_1_identityї
#sequential/random_crop/cond_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/random_crop/cond_1/ConstФ
&sequential/random_crop/cond_1/IdentityIdentity,sequential/random_crop/cond_1/Const:output:0*
T0*
_output_shapes
: 2(
&sequential/random_crop/cond_1/Identity"Y
&sequential_random_crop_cond_1_identity/sequential/random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
з	
┘
@__inference_dense_layer_call_and_return_conditional_losses_26010

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└(ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
═

▄
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25903

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╦
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_26100

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_25956

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
Ў

┴
*__inference_sequential_layer_call_fn_26524
random_crop_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallrandom_crop_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_264932
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:           ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:           
+
_user_specified_namerandom_crop_input
┼c
­
E__inference_sequential_layer_call_and_return_conditional_losses_26493

inputs
conv2d_26453
conv2d_26455
conv2d_1_26458
conv2d_1_26460
batch_normalization_26466
batch_normalization_26468
batch_normalization_26470
batch_normalization_26472
dense_26475
dense_26477
dense_1_26481
dense_1_26483
dense_2_26487
dense_2_26489
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCall\
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:2
random_crop/Shapeї
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
random_crop/strided_slice/stackљ
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_1љ
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_2ф
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_sliceљ
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice_1/stackћ
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_1ћ
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_2┤
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_slice_1љ
random_crop/truediv/CastCast"random_crop/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv/Castќ
random_crop/truediv/Cast_1Cast$random_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv/Cast_1ћ
random_crop/truedivRealDivrandom_crop/truediv/Cast:y:0random_crop/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truedivw
random_crop/Greater/yConst*
_output_shapes
: *
dtype0*
valueB 2      ­?2
random_crop/Greater/yЈ
random_crop/GreaterGreaterrandom_crop/truediv:z:0random_crop/Greater/y:output:0*
T0*
_output_shapes
: 2
random_crop/Greaterћ
random_crop/condStatelessIfrandom_crop/Greater:z:0"random_crop/strided_slice:output:0$random_crop/strided_slice_1:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 */
else_branch R
random_crop_cond_false_26395*
output_shapes
: *.
then_branchR
random_crop_cond_true_263942
random_crop/cond~
random_crop/cond/IdentityIdentityrandom_crop/cond:output:0*
T0*
_output_shapes
: 2
random_crop/cond/Identityю
random_crop/cond_1StatelessIfrandom_crop/Greater:z:0$random_crop/strided_slice_1:output:0"random_crop/strided_slice:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *1
else_branch"R 
random_crop_cond_1_false_26414*
output_shapes
: *0
then_branch!R
random_crop_cond_1_true_264132
random_crop/cond_1ё
random_crop/cond_1/IdentityIdentityrandom_crop/cond_1:output:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identityд
random_crop/stackPack"random_crop/cond/Identity:output:0$random_crop/cond_1/Identity:output:0*
N*
T0*
_output_shapes
:2
random_crop/stackР
!random_crop/resize/ResizeBilinearResizeBilinearinputsrandom_crop/stack:output:0*
T0*A
_output_shapes/
-:+                           *
half_pixel_centers(2#
!random_crop/resize/ResizeBilinearh
random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/sub/yі
random_crop/subSub"random_crop/cond/Identity:output:0random_crop/sub/y:output:0*
T0*
_output_shapes
: 2
random_crop/subl
random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/sub_1/yњ
random_crop/sub_1Sub$random_crop/cond_1/Identity:output:0random_crop/sub_1/y:output:0*
T0*
_output_shapes
: 2
random_crop/sub_1t
random_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/truediv_1/yЁ
random_crop/truediv_1/CastCastrandom_crop/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_1/Castќ
random_crop/truediv_1/Cast_1Cast random_crop/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_1/Cast_1ю
random_crop/truediv_1RealDivrandom_crop/truediv_1/Cast:y:0 random_crop/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truediv_1w
random_crop/CastCastrandom_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/Castt
random_crop/truediv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/truediv_2/yЄ
random_crop/truediv_2/CastCastrandom_crop/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_2/Castќ
random_crop/truediv_2/Cast_1Cast random_crop/truediv_2/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_2/Cast_1ю
random_crop/truediv_2RealDivrandom_crop/truediv_2/Cast:y:0 random_crop/truediv_2/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truediv_2{
random_crop/Cast_1Castrandom_crop/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/Cast_1p
random_crop/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
random_crop/stack_1/0p
random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
value	B : 2
random_crop/stack_1/3╬
random_crop/stack_1Packrandom_crop/stack_1/0:output:0random_crop/Cast:y:0random_crop/Cast_1:y:0random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:2
random_crop/stack_1Ѓ
random_crop/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"              2
random_crop/stack_2Т
random_crop/SliceSlice2random_crop/resize/ResizeBilinear:resized_images:0random_crop/stack_1:output:0random_crop/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         2
random_crop/Sliceд
conv2d/StatefulPartitionedCallStatefulPartitionedCallrandom_crop/Slice:output:0conv2d_26453conv2d_26455*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_258762 
conv2d/StatefulPartitionedCallй
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_26458conv2d_1_26460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_259032"
 conv2d_1/StatefulPartitionedCallљ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_256762
max_pooling2d/PartitionedCallч
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_259372
dropout/PartitionedCallЬ
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_259562
flatten/PartitionedCallа
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_26466batch_normalization_26468batch_normalization_26470batch_normalization_26472*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_258112-
+batch_normalization/StatefulPartitionedCall┤
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26475dense_26477*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_260102
dense/StatefulPartitionedCallЩ
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_260432
dropout_1/PartitionedCallг
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_26481dense_1_26483*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_260672!
dense_1/StatefulPartitionedCallЧ
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_261002
dropout_2/PartitionedCallФ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_26487dense_2_26489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_261242!
dense_2/StatefulPartitionedCallм
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:           ::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
ъl
м
E__inference_sequential_layer_call_and_return_conditional_losses_26342

inputs2
.random_crop_stateful_uniform_full_int_resource
conv2d_26302
conv2d_26304
conv2d_1_26307
conv2d_1_26309
batch_normalization_26315
batch_normalization_26317
batch_normalization_26319
batch_normalization_26321
dense_26324
dense_26326
dense_1_26330
dense_1_26332
dense_2_26336
dense_2_26338
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallбrandom_crop/Assert/Assertб%random_crop/stateful_uniform_full_int\
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:2
random_crop/Shapeї
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
random_crop/strided_slice/stackљ
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_1љ
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_2ф
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_sliceљ
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice_1/stackћ
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_1ћ
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_2┤
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_slice_1l
random_crop/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/stack/1l
random_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/stack/2Р
random_crop/stackPack"random_crop/strided_slice:output:0random_crop/stack/1:output:0random_crop/stack/2:output:0$random_crop/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_crop/stackА
random_crop/GreaterEqualGreaterEqualrandom_crop/Shape:output:0random_crop/stack:output:0*
T0*
_output_shapes
:2
random_crop/GreaterEqualp
random_crop/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
random_crop/Const{
random_crop/AllAllrandom_crop/GreaterEqual:z:0random_crop/Const:output:0*
_output_shapes
: 2
random_crop/Allv
random_crop/Assert/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/Assert/Constz
random_crop/Assert/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/Assert/Const_1є
 random_crop/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 random_crop/Assert/Assert/data_0є
 random_crop/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 random_crop/Assert/Assert/data_1М
random_crop/Assert/AssertAssertrandom_crop/All:output:0)random_crop/Assert/Assert/data_0:output:0)random_crop/Assert/Assert/data_1:output:0*
T
2*
_output_shapes
 2
random_crop/Assert/Assert¤
random_crop/control_dependencyIdentityrandom_crop/Shape:output:0^random_crop/Assert/Assert*
T0*$
_class
loc:@random_crop/Shape*
_output_shapes
:2 
random_crop/control_dependencyЊ
random_crop/subSub'random_crop/control_dependency:output:0random_crop/stack:output:0*
T0*
_output_shapes
:2
random_crop/subh
random_crop/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/add/yЂ
random_crop/addAddV2random_crop/sub:z:0random_crop/add/y:output:0*
T0*
_output_shapes
:2
random_crop/addt
random_crop/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
random_crop/Shape_1ц
+random_crop/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_crop/stateful_uniform_full_int/shapeц
/random_crop/stateful_uniform_full_int/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/random_crop/stateful_uniform_full_int/algorithm─
%random_crop/stateful_uniform_full_intStatefulUniformFullInt.random_crop_stateful_uniform_full_int_resource8random_crop/stateful_uniform_full_int/algorithm:output:04random_crop/stateful_uniform_full_int/shape:output:0*
_output_shapes
:*
dtype0	*
shape_dtype02'
%random_crop/stateful_uniform_full_intz
random_crop/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_crop/zeros_likeх
random_crop/stack_1Pack.random_crop/stateful_uniform_full_int:output:0random_crop/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_crop/stack_1Ќ
!random_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_crop/strided_slice_2/stackЏ
#random_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_crop/strided_slice_2/stack_1Џ
#random_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_crop/strided_slice_2/stack_2▄
random_crop/strided_slice_2StridedSlicerandom_crop/stack_1:output:0*random_crop/strided_slice_2/stack:output:0,random_crop/strided_slice_2/stack_1:output:0,random_crop/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_crop/strided_slice_2ќ
(random_crop/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2*
(random_crop/stateless_random_uniform/minџ
(random_crop/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :    2*
(random_crop/stateless_random_uniform/max┼
$random_crop/stateless_random_uniformStatelessRandomUniformIntrandom_crop/Shape_1:output:0$random_crop/strided_slice_2:output:01random_crop/stateless_random_uniform/min:output:01random_crop/stateless_random_uniform/max:output:0*
T0*
_output_shapes
:*
dtype02&
$random_crop/stateless_random_uniformЌ
random_crop/modFloorMod-random_crop/stateless_random_uniform:output:0random_crop/add:z:0*
T0*
_output_shapes
:2
random_crop/mod»
random_crop/SliceSliceinputsrandom_crop/mod:z:0random_crop/stack:output:0*
Index0*
T0*/
_output_shapes
:         2
random_crop/Sliceд
conv2d/StatefulPartitionedCallStatefulPartitionedCallrandom_crop/Slice:output:0conv2d_26302conv2d_26304*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_258762 
conv2d/StatefulPartitionedCallй
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_26307conv2d_1_26309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_259032"
 conv2d_1/StatefulPartitionedCallљ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_256762
max_pooling2d/PartitionedCall»
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0^random_crop/Assert/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_259322!
dropout/StatefulPartitionedCallШ
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_259562
flatten/PartitionedCallъ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_26315batch_normalization_26317batch_normalization_26319batch_normalization_26321*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_257782-
+batch_normalization/StatefulPartitionedCall┤
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26324dense_26326*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_260102
dense/StatefulPartitionedCall┤
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_260382#
!dropout_1/StatefulPartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_26330dense_1_26332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_260672!
dense_1/StatefulPartitionedCallИ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_260952#
!dropout_2/StatefulPartitionedCall│
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_26336dense_2_26338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_261242!
dense_2/StatefulPartitionedCallђ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^random_crop/Assert/Assert&^random_crop/stateful_uniform_full_int*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:           :::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall26
random_crop/Assert/Assertrandom_crop/Assert/Assert2N
%random_crop/stateful_uniform_full_int%random_crop/stateful_uniform_full_int:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
И

Л
*__inference_sequential_layer_call_fn_26375
random_crop_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallrandom_crop_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_263422
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:           :::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:           
+
_user_specified_namerandom_crop_input
ф
I
-__inference_max_pooling2d_layer_call_fn_25682

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_256762
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ќ
┘
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25811

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└(2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└(2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:└(2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└(2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└(2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └(::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
ш	
█
B__inference_dense_1_layer_call_and_return_conditional_losses_27122

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
т

▒
random_crop_cond_true_263942
.random_crop_cond_mul_random_crop_strided_slice?
;random_crop_cond_truediv_cast_1_random_crop_strided_slice_1
random_crop_cond_identityr
random_crop/cond/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond/mul/xЦ
random_crop/cond/mulMulrandom_crop/cond/mul/x:output:0.random_crop_cond_mul_random_crop_strided_slice*
T0*
_output_shapes
: 2
random_crop/cond/mulљ
random_crop/cond/truediv/CastCastrandom_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond/truediv/Castи
random_crop/cond/truediv/Cast_1Cast;random_crop_cond_truediv_cast_1_random_crop_strided_slice_1*

DstT0*

SrcT0*
_output_shapes
: 2!
random_crop/cond/truediv/Cast_1е
random_crop/cond/truedivRealDiv!random_crop/cond/truediv/Cast:y:0#random_crop/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/cond/truedivё
random_crop/cond/CastCastrandom_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond/Cast~
random_crop/cond/IdentityIdentityrandom_crop/cond/Cast:y:0*
T0*
_output_shapes
: 2
random_crop/cond/Identity"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
§
}
(__inference_conv2d_1_layer_call_fn_26944

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_259032
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
■
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25676

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ќ
┘
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27038

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:└(*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└(2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└(2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:└(2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└(2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└(2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └(2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └(::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
ѓк
└
!__inference__traced_restore_27498
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance#
assignvariableop_8_dense_kernel!
assignvariableop_9_dense_bias&
"assignvariableop_10_dense_1_kernel$
 assignvariableop_11_dense_1_bias&
"assignvariableop_12_dense_2_kernel$
 assignvariableop_13_dense_2_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay 
assignvariableop_18_variable
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_1,
(assignvariableop_23_adam_conv2d_kernel_m*
&assignvariableop_24_adam_conv2d_bias_m.
*assignvariableop_25_adam_conv2d_1_kernel_m,
(assignvariableop_26_adam_conv2d_1_bias_m8
4assignvariableop_27_adam_batch_normalization_gamma_m7
3assignvariableop_28_adam_batch_normalization_beta_m+
'assignvariableop_29_adam_dense_kernel_m)
%assignvariableop_30_adam_dense_bias_m-
)assignvariableop_31_adam_dense_1_kernel_m+
'assignvariableop_32_adam_dense_1_bias_m-
)assignvariableop_33_adam_dense_2_kernel_m+
'assignvariableop_34_adam_dense_2_bias_m,
(assignvariableop_35_adam_conv2d_kernel_v*
&assignvariableop_36_adam_conv2d_bias_v.
*assignvariableop_37_adam_conv2d_1_kernel_v,
(assignvariableop_38_adam_conv2d_1_bias_v8
4assignvariableop_39_adam_batch_normalization_gamma_v7
3assignvariableop_40_adam_batch_normalization_beta_v+
'assignvariableop_41_adam_dense_kernel_v)
%assignvariableop_42_adam_dense_bias_v-
)assignvariableop_43_adam_dense_1_kernel_v+
'assignvariableop_44_adam_dense_1_bias_v-
)assignvariableop_45_adam_dense_2_kernel_v+
'assignvariableop_46_adam_dense_2_bias_v
identity_48ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9╗
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*К
valueйB║0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesъ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*о
_output_shapes├
└::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Б
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Д
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5░
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6и
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╗
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ц
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9б
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ф
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11е
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ф
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13е
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14Ц
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Д
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Д
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17д
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18ц
AssignVariableOp_18AssignVariableOpassignvariableop_18_variableIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19А
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20А
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Б
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Б
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23░
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╝
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_batch_normalization_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╗
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_batch_normalization_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29»
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Г
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31▒
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32»
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33▒
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34»
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35░
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv2d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36«
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv2d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39╝
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_batch_normalization_gamma_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40╗
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_batch_normalization_beta_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41»
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Г
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_dense_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▒
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44»
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45▒
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46»
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_469
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpУ
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47█
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*М
_input_shapes┴
Й: :::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
й
`
'__inference_dropout_layer_call_fn_26966

inputs
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_259322
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         		@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         		@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
╣Ѓ
Ж	
E__inference_sequential_layer_call_and_return_conditional_losses_26836

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource4
0batch_normalization_cast_readvariableop_resource6
2batch_normalization_cast_1_readvariableop_resource6
2batch_normalization_cast_2_readvariableop_resource6
2batch_normalization_cast_3_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityѕб'batch_normalization/Cast/ReadVariableOpб)batch_normalization/Cast_1/ReadVariableOpб)batch_normalization/Cast_2/ReadVariableOpб)batch_normalization/Cast_3/ReadVariableOpбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOp\
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:2
random_crop/Shapeї
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
random_crop/strided_slice/stackљ
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_1љ
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_2ф
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_sliceљ
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice_1/stackћ
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_1ћ
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_2┤
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_slice_1љ
random_crop/truediv/CastCast"random_crop/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv/Castќ
random_crop/truediv/Cast_1Cast$random_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv/Cast_1ћ
random_crop/truedivRealDivrandom_crop/truediv/Cast:y:0random_crop/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truedivw
random_crop/Greater/yConst*
_output_shapes
: *
dtype0*
valueB 2      ­?2
random_crop/Greater/yЈ
random_crop/GreaterGreaterrandom_crop/truediv:z:0random_crop/Greater/y:output:0*
T0*
_output_shapes
: 2
random_crop/Greaterћ
random_crop/condStatelessIfrandom_crop/Greater:z:0"random_crop/strided_slice:output:0$random_crop/strided_slice_1:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 */
else_branch R
random_crop_cond_false_26720*
output_shapes
: *.
then_branchR
random_crop_cond_true_267192
random_crop/cond~
random_crop/cond/IdentityIdentityrandom_crop/cond:output:0*
T0*
_output_shapes
: 2
random_crop/cond/Identityю
random_crop/cond_1StatelessIfrandom_crop/Greater:z:0$random_crop/strided_slice_1:output:0"random_crop/strided_slice:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *1
else_branch"R 
random_crop_cond_1_false_26739*
output_shapes
: *0
then_branch!R
random_crop_cond_1_true_267382
random_crop/cond_1ё
random_crop/cond_1/IdentityIdentityrandom_crop/cond_1:output:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identityд
random_crop/stackPack"random_crop/cond/Identity:output:0$random_crop/cond_1/Identity:output:0*
N*
T0*
_output_shapes
:2
random_crop/stackР
!random_crop/resize/ResizeBilinearResizeBilinearinputsrandom_crop/stack:output:0*
T0*A
_output_shapes/
-:+                           *
half_pixel_centers(2#
!random_crop/resize/ResizeBilinearh
random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/sub/yі
random_crop/subSub"random_crop/cond/Identity:output:0random_crop/sub/y:output:0*
T0*
_output_shapes
: 2
random_crop/subl
random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/sub_1/yњ
random_crop/sub_1Sub$random_crop/cond_1/Identity:output:0random_crop/sub_1/y:output:0*
T0*
_output_shapes
: 2
random_crop/sub_1t
random_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/truediv_1/yЁ
random_crop/truediv_1/CastCastrandom_crop/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_1/Castќ
random_crop/truediv_1/Cast_1Cast random_crop/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_1/Cast_1ю
random_crop/truediv_1RealDivrandom_crop/truediv_1/Cast:y:0 random_crop/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truediv_1w
random_crop/CastCastrandom_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/Castt
random_crop/truediv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/truediv_2/yЄ
random_crop/truediv_2/CastCastrandom_crop/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_2/Castќ
random_crop/truediv_2/Cast_1Cast random_crop/truediv_2/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/truediv_2/Cast_1ю
random_crop/truediv_2RealDivrandom_crop/truediv_2/Cast:y:0 random_crop/truediv_2/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/truediv_2{
random_crop/Cast_1Castrandom_crop/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/Cast_1p
random_crop/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
random_crop/stack_1/0p
random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
value	B : 2
random_crop/stack_1/3╬
random_crop/stack_1Packrandom_crop/stack_1/0:output:0random_crop/Cast:y:0random_crop/Cast_1:y:0random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:2
random_crop/stack_1Ѓ
random_crop/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"              2
random_crop/stack_2Т
random_crop/SliceSlice2random_crop/resize/ResizeBilinear:resized_images:0random_crop/stack_1:output:0random_crop/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         2
random_crop/Sliceф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╠
conv2d/Conv2DConv2Drandom_crop/Slice:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЛ
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu├
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         		@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolі
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         		@2
dropout/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
flatten/ConstЊ
flatten/ReshapeReshapedropout/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         └(2
flatten/Reshape└
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:└(*
dtype02)
'batch_normalization/Cast/ReadVariableOpк
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:└(*
dtype02+
)batch_normalization/Cast_1/ReadVariableOpк
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:└(*
dtype02+
)batch_normalization/Cast_2/ReadVariableOpк
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:└(*
dtype02+
)batch_normalization/Cast_3/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yо
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└(2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:└(2%
#batch_normalization/batchnorm/Rsqrt¤
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:└(2#
!batch_normalization/batchnorm/mul┼
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         └(2%
#batch_normalization/batchnorm/mul_1¤
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:└(2%
#batch_normalization/batchnorm/mul_2¤
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└(2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         └(2%
#batch_normalization/batchnorm/add_1А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
└(ђ*
dtype02
dense/MatMul/ReadVariableOpД
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2

dense/ReluЂ
dropout_1/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_1/IdentityД
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_1/MatMul/ReadVariableOpА
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_1/ReluЃ
dropout_2/IdentityIdentitydense_1/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_2/Identityд
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
dense_2/MatMul/ReadVariableOpа
dense_2/MatMulMatMuldropout_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_2/Softmax▄
IdentityIdentitydense_2/Softmax:softmax:0(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:           ::::::::::::::2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
щ╩
▄
E__inference_sequential_layer_call_and_return_conditional_losses_26702

inputs2
.random_crop_stateful_uniform_full_int_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource-
)batch_normalization_assignmovingavg_26640/
+batch_normalization_assignmovingavg_1_266464
0batch_normalization_cast_readvariableop_resource6
2batch_normalization_cast_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityѕб7batch_normalization/AssignMovingAvg/AssignSubVariableOpб2batch_normalization/AssignMovingAvg/ReadVariableOpб9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpб4batch_normalization/AssignMovingAvg_1/ReadVariableOpб'batch_normalization/Cast/ReadVariableOpб)batch_normalization/Cast_1/ReadVariableOpбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбrandom_crop/Assert/Assertб%random_crop/stateful_uniform_full_int\
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:2
random_crop/Shapeї
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
random_crop/strided_slice/stackљ
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_1љ
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice/stack_2ф
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_sliceљ
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_crop/strided_slice_1/stackћ
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_1ћ
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_crop/strided_slice_1/stack_2┤
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_crop/strided_slice_1l
random_crop/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/stack/1l
random_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/stack/2Р
random_crop/stackPack"random_crop/strided_slice:output:0random_crop/stack/1:output:0random_crop/stack/2:output:0$random_crop/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_crop/stackА
random_crop/GreaterEqualGreaterEqualrandom_crop/Shape:output:0random_crop/stack:output:0*
T0*
_output_shapes
:2
random_crop/GreaterEqualp
random_crop/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
random_crop/Const{
random_crop/AllAllrandom_crop/GreaterEqual:z:0random_crop/Const:output:0*
_output_shapes
: 2
random_crop/Allv
random_crop/Assert/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/Assert/Constz
random_crop/Assert/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
random_crop/Assert/Const_1є
 random_crop/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 random_crop/Assert/Assert/data_0є
 random_crop/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 random_crop/Assert/Assert/data_1М
random_crop/Assert/AssertAssertrandom_crop/All:output:0)random_crop/Assert/Assert/data_0:output:0)random_crop/Assert/Assert/data_1:output:0*
T
2*
_output_shapes
 2
random_crop/Assert/Assert¤
random_crop/control_dependencyIdentityrandom_crop/Shape:output:0^random_crop/Assert/Assert*
T0*$
_class
loc:@random_crop/Shape*
_output_shapes
:2 
random_crop/control_dependencyЊ
random_crop/subSub'random_crop/control_dependency:output:0random_crop/stack:output:0*
T0*
_output_shapes
:2
random_crop/subh
random_crop/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/add/yЂ
random_crop/addAddV2random_crop/sub:z:0random_crop/add/y:output:0*
T0*
_output_shapes
:2
random_crop/addt
random_crop/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
random_crop/Shape_1ц
+random_crop/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_crop/stateful_uniform_full_int/shapeц
/random_crop/stateful_uniform_full_int/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/random_crop/stateful_uniform_full_int/algorithm─
%random_crop/stateful_uniform_full_intStatefulUniformFullInt.random_crop_stateful_uniform_full_int_resource8random_crop/stateful_uniform_full_int/algorithm:output:04random_crop/stateful_uniform_full_int/shape:output:0*
_output_shapes
:*
dtype0	*
shape_dtype02'
%random_crop/stateful_uniform_full_intz
random_crop/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_crop/zeros_likeх
random_crop/stack_1Pack.random_crop/stateful_uniform_full_int:output:0random_crop/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_crop/stack_1Ќ
!random_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_crop/strided_slice_2/stackЏ
#random_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_crop/strided_slice_2/stack_1Џ
#random_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_crop/strided_slice_2/stack_2▄
random_crop/strided_slice_2StridedSlicerandom_crop/stack_1:output:0*random_crop/strided_slice_2/stack:output:0,random_crop/strided_slice_2/stack_1:output:0,random_crop/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_crop/strided_slice_2ќ
(random_crop/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2*
(random_crop/stateless_random_uniform/minџ
(random_crop/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :    2*
(random_crop/stateless_random_uniform/max┼
$random_crop/stateless_random_uniformStatelessRandomUniformIntrandom_crop/Shape_1:output:0$random_crop/strided_slice_2:output:01random_crop/stateless_random_uniform/min:output:01random_crop/stateless_random_uniform/max:output:0*
T0*
_output_shapes
:*
dtype02&
$random_crop/stateless_random_uniformЌ
random_crop/modFloorMod-random_crop/stateless_random_uniform:output:0random_crop/add:z:0*
T0*
_output_shapes
:2
random_crop/mod»
random_crop/SliceSliceinputsrandom_crop/mod:z:0random_crop/stack:output:0*
Index0*
T0*/
_output_shapes
:         2
random_crop/Sliceф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╠
conv2d/Conv2DConv2Drandom_crop/Slice:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЛ
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu├
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         		@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/dropout/ConstФ
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         		@2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeр
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         		@*
dtype0*
seedм2.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2 
dropout/dropout/GreaterEqual/yТ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         		@2
dropout/dropout/GreaterEqualЪ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         		@2
dropout/dropout/Castб
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         		@2
dropout/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
flatten/ConstЊ
flatten/ReshapeReshapedropout/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:         └(2
flatten/Reshape▓
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indicesя
 batch_normalization/moments/meanMeanflatten/Reshape:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	└(*
	keep_dims(2"
 batch_normalization/moments/mean╣
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	└(2*
(batch_normalization/moments/StopGradientз
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:         └(2/
-batch_normalization/moments/SquaredDifference║
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indicesЃ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	└(*
	keep_dims(2&
$batch_normalization/moments/varianceй
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:└(*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:└(*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Є
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/26640*
_output_shapes
: *
dtype0*
valueB
 *
О#<2+
)batch_normalization/AssignMovingAvg/decay¤
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_26640*
_output_shapes	
:└(*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpН
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/26640*
_output_shapes	
:└(2)
'batch_normalization/AssignMovingAvg/sub╠
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/26640*
_output_shapes	
:└(2)
'batch_normalization/AssignMovingAvg/mulЦ
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_26640+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/26640*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpЇ
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/26646*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization/AssignMovingAvg_1/decayН
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_26646*
_output_shapes	
:└(*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp▀
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/26646*
_output_shapes	
:└(2+
)batch_normalization/AssignMovingAvg_1/subо
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/26646*
_output_shapes	
:└(2+
)batch_normalization/AssignMovingAvg_1/mul▒
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_26646-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/26646*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp└
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:└(*
dtype02)
'batch_normalization/Cast/ReadVariableOpк
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:└(*
dtype02+
)batch_normalization/Cast_1/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yМ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└(2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:└(2%
#batch_normalization/batchnorm/Rsqrt¤
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:└(2#
!batch_normalization/batchnorm/mul┼
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         └(2%
#batch_normalization/batchnorm/mul_1╠
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:└(2%
#batch_normalization/batchnorm/mul_2═
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└(2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         └(2%
#batch_normalization/batchnorm/add_1А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
└(ђ*
dtype02
dense/MatMul/ReadVariableOpД
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2

dense/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout_1/dropout/Constц
dropout_1/dropout/MulMuldense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeь
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedм*
seed220
.dropout_1/dropout/random_uniform/RandomUniformЅ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2"
 dropout_1/dropout/GreaterEqual/yу
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2 
dropout_1/dropout/GreaterEqualъ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_1/dropout/CastБ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_1/dropout/Mul_1Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_1/MatMul/ReadVariableOpА
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout_2/dropout/Constд
dropout_2/dropout/MulMuldense_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_2/dropout/Mul|
dropout_2/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeь
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedм*
seed220
.dropout_2/dropout/random_uniform/RandomUniformЅ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2"
 dropout_2/dropout/GreaterEqual/yу
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2 
dropout_2/dropout/GreaterEqualъ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_2/dropout/CastБ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_2/dropout/Mul_1д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
dense_2/MatMul/ReadVariableOpа
dense_2/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_2/Softmaxф
IdentityIdentitydense_2/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^random_crop/Assert/Assert&^random_crop/stateful_uniform_full_int*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:           :::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp26
random_crop/Assert/Assertrandom_crop/Assert/Assert2N
%random_crop/stateful_uniform_full_int%random_crop/stateful_uniform_full_int:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
ќ
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_27096

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedм2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ъ
║
random_crop_cond_1_false_264146
2random_crop_cond_1_mul_random_crop_strided_slice_1?
;random_crop_cond_1_truediv_cast_1_random_crop_strided_slice
random_crop_cond_1_identityv
random_crop/cond_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond_1/mul/x»
random_crop/cond_1/mulMul!random_crop/cond_1/mul/x:output:02random_crop_cond_1_mul_random_crop_strided_slice_1*
T0*
_output_shapes
: 2
random_crop/cond_1/mulќ
random_crop/cond_1/truediv/CastCastrandom_crop/cond_1/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2!
random_crop/cond_1/truediv/Cast╗
!random_crop/cond_1/truediv/Cast_1Cast;random_crop_cond_1_truediv_cast_1_random_crop_strided_slice*

DstT0*

SrcT0*
_output_shapes
: 2#
!random_crop/cond_1/truediv/Cast_1░
random_crop/cond_1/truedivRealDiv#random_crop/cond_1/truediv/Cast:y:0%random_crop/cond_1/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2
random_crop/cond_1/truedivі
random_crop/cond_1/CastCastrandom_crop/cond_1/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_crop/cond_1/Castё
random_crop/cond_1/IdentityIdentityrandom_crop/cond_1/Cast:y:0*
T0*
_output_shapes
: 2
random_crop/cond_1/Identity"C
random_crop_cond_1_identity$random_crop/cond_1/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
Ў
E
)__inference_dropout_1_layer_call_fn_27111

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_260432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╦
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_27101

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_26977

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*.
_input_shapes
:         		@:W S
/
_output_shapes
:         		@
 
_user_specified_nameinputs
Ц
b
)__inference_dropout_2_layer_call_fn_27153

inputs
identityѕбStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_260952
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_26095

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedм2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ
Ѓ
random_crop_cond_false_26720 
random_crop_cond_placeholder"
random_crop_cond_placeholder_1
random_crop_cond_identityr
random_crop/cond/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
random_crop/cond/Constё
random_crop/cond/IdentityIdentityrandom_crop/cond/Const:output:0*
T0*
_output_shapes
: 2
random_crop/cond/Identity"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
Х
д
3__inference_batch_normalization_layer_call_fn_27051

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_257782
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └(::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └(
 
_user_specified_nameinputs
їА
ё
 __inference__wrapped_model_25670
random_crop_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource?
;sequential_batch_normalization_cast_readvariableop_resourceA
=sequential_batch_normalization_cast_1_readvariableop_resourceA
=sequential_batch_normalization_cast_2_readvariableop_resourceA
=sequential_batch_normalization_cast_3_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identityѕб2sequential/batch_normalization/Cast/ReadVariableOpб4sequential/batch_normalization/Cast_1/ReadVariableOpб4sequential/batch_normalization/Cast_2/ReadVariableOpб4sequential/batch_normalization/Cast_3/ReadVariableOpб(sequential/conv2d/BiasAdd/ReadVariableOpб'sequential/conv2d/Conv2D/ReadVariableOpб*sequential/conv2d_1/BiasAdd/ReadVariableOpб)sequential/conv2d_1/Conv2D/ReadVariableOpб'sequential/dense/BiasAdd/ReadVariableOpб&sequential/dense/MatMul/ReadVariableOpб)sequential/dense_1/BiasAdd/ReadVariableOpб(sequential/dense_1/MatMul/ReadVariableOpб)sequential/dense_2/BiasAdd/ReadVariableOpб(sequential/dense_2/MatMul/ReadVariableOp}
sequential/random_crop/ShapeShaperandom_crop_input*
T0*
_output_shapes
:2
sequential/random_crop/Shapeб
*sequential/random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/random_crop/strided_slice/stackд
,sequential/random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_crop/strided_slice/stack_1д
,sequential/random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_crop/strided_slice/stack_2В
$sequential/random_crop/strided_sliceStridedSlice%sequential/random_crop/Shape:output:03sequential/random_crop/strided_slice/stack:output:05sequential/random_crop/strided_slice/stack_1:output:05sequential/random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential/random_crop/strided_sliceд
,sequential/random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_crop/strided_slice_1/stackф
.sequential/random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_crop/strided_slice_1/stack_1ф
.sequential/random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_crop/strided_slice_1/stack_2Ш
&sequential/random_crop/strided_slice_1StridedSlice%sequential/random_crop/Shape:output:05sequential/random_crop/strided_slice_1/stack:output:07sequential/random_crop/strided_slice_1/stack_1:output:07sequential/random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/random_crop/strided_slice_1▒
#sequential/random_crop/truediv/CastCast-sequential/random_crop/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential/random_crop/truediv/Castи
%sequential/random_crop/truediv/Cast_1Cast/sequential/random_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential/random_crop/truediv/Cast_1└
sequential/random_crop/truedivRealDiv'sequential/random_crop/truediv/Cast:y:0)sequential/random_crop/truediv/Cast_1:y:0*
T0*
_output_shapes
: 2 
sequential/random_crop/truedivЇ
 sequential/random_crop/Greater/yConst*
_output_shapes
: *
dtype0*
valueB 2      ­?2"
 sequential/random_crop/Greater/y╗
sequential/random_crop/GreaterGreater"sequential/random_crop/truediv:z:0)sequential/random_crop/Greater/y:output:0*
T0*
_output_shapes
: 2 
sequential/random_crop/Greaterр
sequential/random_crop/condStatelessIf"sequential/random_crop/Greater:z:0-sequential/random_crop/strided_slice:output:0/sequential/random_crop/strided_slice_1:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *:
else_branch+R)
'sequential_random_crop_cond_false_25554*
output_shapes
: *9
then_branch*R(
&sequential_random_crop_cond_true_255532
sequential/random_crop/condЪ
$sequential/random_crop/cond/IdentityIdentity$sequential/random_crop/cond:output:0*
T0*
_output_shapes
: 2&
$sequential/random_crop/cond/Identityж
sequential/random_crop/cond_1StatelessIf"sequential/random_crop/Greater:z:0/sequential/random_crop/strided_slice_1:output:0-sequential/random_crop/strided_slice:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *<
else_branch-R+
)sequential_random_crop_cond_1_false_25573*
output_shapes
: *;
then_branch,R*
(sequential_random_crop_cond_1_true_255722
sequential/random_crop/cond_1Ц
&sequential/random_crop/cond_1/IdentityIdentity&sequential/random_crop/cond_1:output:0*
T0*
_output_shapes
: 2(
&sequential/random_crop/cond_1/Identityм
sequential/random_crop/stackPack-sequential/random_crop/cond/Identity:output:0/sequential/random_crop/cond_1/Identity:output:0*
N*
T0*
_output_shapes
:2
sequential/random_crop/stackј
,sequential/random_crop/resize/ResizeBilinearResizeBilinearrandom_crop_input%sequential/random_crop/stack:output:0*
T0*A
_output_shapes/
-:+                           *
half_pixel_centers(2.
,sequential/random_crop/resize/ResizeBilinear~
sequential/random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/random_crop/sub/yХ
sequential/random_crop/subSub-sequential/random_crop/cond/Identity:output:0%sequential/random_crop/sub/y:output:0*
T0*
_output_shapes
: 2
sequential/random_crop/subѓ
sequential/random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential/random_crop/sub_1/yЙ
sequential/random_crop/sub_1Sub/sequential/random_crop/cond_1/Identity:output:0'sequential/random_crop/sub_1/y:output:0*
T0*
_output_shapes
: 2
sequential/random_crop/sub_1і
"sequential/random_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/random_crop/truediv_1/yд
%sequential/random_crop/truediv_1/CastCastsequential/random_crop/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential/random_crop/truediv_1/Castи
'sequential/random_crop/truediv_1/Cast_1Cast+sequential/random_crop/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'sequential/random_crop/truediv_1/Cast_1╚
 sequential/random_crop/truediv_1RealDiv)sequential/random_crop/truediv_1/Cast:y:0+sequential/random_crop/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: 2"
 sequential/random_crop/truediv_1ў
sequential/random_crop/CastCast$sequential/random_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
sequential/random_crop/Castі
"sequential/random_crop/truediv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/random_crop/truediv_2/yе
%sequential/random_crop/truediv_2/CastCast sequential/random_crop/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential/random_crop/truediv_2/Castи
'sequential/random_crop/truediv_2/Cast_1Cast+sequential/random_crop/truediv_2/y:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'sequential/random_crop/truediv_2/Cast_1╚
 sequential/random_crop/truediv_2RealDiv)sequential/random_crop/truediv_2/Cast:y:0+sequential/random_crop/truediv_2/Cast_1:y:0*
T0*
_output_shapes
: 2"
 sequential/random_crop/truediv_2ю
sequential/random_crop/Cast_1Cast$sequential/random_crop/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
sequential/random_crop/Cast_1є
 sequential/random_crop/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential/random_crop/stack_1/0є
 sequential/random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential/random_crop/stack_1/3љ
sequential/random_crop/stack_1Pack)sequential/random_crop/stack_1/0:output:0sequential/random_crop/Cast:y:0!sequential/random_crop/Cast_1:y:0)sequential/random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:2 
sequential/random_crop/stack_1Ў
sequential/random_crop/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"              2 
sequential/random_crop/stack_2Ю
sequential/random_crop/SliceSlice=sequential/random_crop/resize/ResizeBilinear:resized_images:0'sequential/random_crop/stack_1:output:0'sequential/random_crop/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         2
sequential/random_crop/Slice╦
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpЭ
sequential/conv2d/Conv2DConv2D%sequential/random_crop/Slice:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential/conv2d/Conv2D┬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpл
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
sequential/conv2d/BiasAddќ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential/conv2d/ReluЛ
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp§
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D╚
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpп
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
sequential/conv2d_1/BiasAddю
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
sequential/conv2d_1/ReluС
 sequential/max_pooling2d/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:         		@*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolФ
sequential/dropout/IdentityIdentity)sequential/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         		@2
sequential/dropout/IdentityЁ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
sequential/flatten/Const┐
sequential/flatten/ReshapeReshape$sequential/dropout/Identity:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:         └(2
sequential/flatten/Reshapeр
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:└(*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOpу
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:└(*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOpу
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:└(*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOpу
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:└(*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOpЦ
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:20
.sequential/batch_normalization/batchnorm/add/yѓ
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└(2.
,sequential/batch_normalization/batchnorm/add┴
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:└(20
.sequential/batch_normalization/batchnorm/Rsqrtч
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:└(2.
,sequential/batch_normalization/batchnorm/mulы
.sequential/batch_normalization/batchnorm/mul_1Mul#sequential/flatten/Reshape:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         └(20
.sequential/batch_normalization/batchnorm/mul_1ч
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:└(20
.sequential/batch_normalization/batchnorm/mul_2ч
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└(2.
,sequential/batch_normalization/batchnorm/subѓ
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         └(20
.sequential/batch_normalization/batchnorm/add_1┬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
└(ђ*
dtype02(
&sequential/dense/MatMul/ReadVariableOpМ
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/MatMul└
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpк
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/BiasAddї
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/Reluб
sequential/dropout_1/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
sequential/dropout_1/Identity╚
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp═
sequential/dense_1/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense_1/MatMulк
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp╬
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense_1/BiasAddњ
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential/dense_1/Reluц
sequential/dropout_2/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
sequential/dropout_2/IdentityК
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp╠
sequential/dense_2/MatMulMatMul&sequential/dropout_2/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential/dense_2/MatMul┼
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp═
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential/dense_2/BiasAddџ
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
sequential/dense_2/SoftmaxЂ
IdentityIdentity$sequential/dense_2/Softmax:softmax:03^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:           ::::::::::::::2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:b ^
/
_output_shapes
:           
+
_user_specified_namerandom_crop_input"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*к
serving_default▓
W
random_crop_inputB
#serving_default_random_crop_input:0           ;
dense_20
StatefulPartitionedCall:0         
tensorflow/serving/predict:ќП
 R
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
┤_default_save_signature
+х&call_and_return_all_conditional_losses
Х__call__"эN
_tf_keras_sequentialпN{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_crop_input"}}, {"class_name": "RandomCrop", "config": {"name": "random_crop", "trainable": true, "dtype": "float32", "height": 29, "width": 29, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_crop_input"}}, {"class_name": "RandomCrop", "config": {"name": "random_crop", "trainable": true, "dtype": "float32", "height": 29, "width": 29, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": {"class_name": "ExponentialDecay", "config": {"initial_learning_rate": 0.001, "decay_steps": 9000, "decay_rate": 0.98, "staircase": true, "name": null}}, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ы
_rng
	keras_api"Н
_tf_keras_layer╗{"class_name": "RandomCrop", "name": "random_crop", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_crop", "trainable": true, "dtype": "float32", "height": 29, "width": 29, "seed": null}}
№


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
и__call__
+И&call_and_return_all_conditional_losses"╚	
_tf_keras_layer«	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 29, 29, 3]}}
З	

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 29, 29, 32]}}
§
!trainable_variables
"regularization_losses
#	variables
$	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"В
_tf_keras_layerм{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
с
%trainable_variables
&regularization_losses
'	variables
(	keras_api
й__call__
+Й&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
С
)trainable_variables
*regularization_losses
+	variables
,	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"М
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
┤	
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2trainable_variables
3regularization_losses
4	variables
5	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"я
_tf_keras_layer─{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 5184}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5184]}}
з

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
├__call__
+─&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5184}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5184]}}
у
<trainable_variables
=regularization_losses
>	variables
?	keras_api
┼__call__
+к&call_and_return_all_conditional_losses"о
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ш

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
К__call__
+╚&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
у
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses"о
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
э

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
░
Piter

Qbeta_1

Rbeta_2
	SdecaymюmЮmъmЪ.mа/mА6mб7mБ@mцAmЦJmдKmДvеvЕvфvФ.vг/vГ6v«7v»@v░Av▒Jv▓Kv│"
	optimizer
v
0
1
2
3
.4
/5
66
77
@8
A9
J10
K11"
trackable_list_wrapper
 "
trackable_list_wrapper
є
0
1
2
3
.4
/5
06
17
68
79
@10
A11
J12
K13"
trackable_list_wrapper
╬
trainable_variables
Tlayer_metrics
Ulayer_regularization_losses
regularization_losses
Vnon_trainable_variables
Wmetrics
	variables

Xlayers
Х__call__
┤_default_save_signature
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
-
═serving_default"
signature_map
.
Y
_state_var"
_generic_user_object
"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
trainable_variables
Zlayer_metrics
[layer_regularization_losses
regularization_losses
\non_trainable_variables
]metrics
	variables

^layers
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
trainable_variables
_layer_metrics
`layer_regularization_losses
regularization_losses
anon_trainable_variables
bmetrics
	variables

clayers
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
!trainable_variables
dlayer_metrics
elayer_regularization_losses
"regularization_losses
fnon_trainable_variables
gmetrics
#	variables

hlayers
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
%trainable_variables
ilayer_metrics
jlayer_regularization_losses
&regularization_losses
knon_trainable_variables
lmetrics
'	variables

mlayers
й__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
)trainable_variables
nlayer_metrics
olayer_regularization_losses
*regularization_losses
pnon_trainable_variables
qmetrics
+	variables

rlayers
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&└(2batch_normalization/gamma
':%└(2batch_normalization/beta
0:.└( (2batch_normalization/moving_mean
4:2└( (2#batch_normalization/moving_variance
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
░
2trainable_variables
slayer_metrics
tlayer_regularization_losses
3regularization_losses
unon_trainable_variables
vmetrics
4	variables

wlayers
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 :
└(ђ2dense/kernel
:ђ2
dense/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
░
8trainable_variables
xlayer_metrics
ylayer_regularization_losses
9regularization_losses
znon_trainable_variables
{metrics
:	variables

|layers
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
<trainable_variables
}layer_metrics
~layer_regularization_losses
=regularization_losses
non_trainable_variables
ђmetrics
>	variables
Ђlayers
┼__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
": 
ђђ2dense_1/kernel
:ђ2dense_1/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
х
Btrainable_variables
ѓlayer_metrics
 Ѓlayer_regularization_losses
Cregularization_losses
ёnon_trainable_variables
Ёmetrics
D	variables
єlayers
К__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ftrainable_variables
Єlayer_metrics
 ѕlayer_regularization_losses
Gregularization_losses
Ѕnon_trainable_variables
іmetrics
H	variables
Іlayers
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
!:	ђ
2dense_2/kernel
:
2dense_2/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
х
Ltrainable_variables
їlayer_metrics
 Їlayer_regularization_losses
Mregularization_losses
јnon_trainable_variables
Јmetrics
N	variables
љlayers
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
0
Љ0
њ1"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
:	2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┐

Њtotal

ћcount
Ћ	variables
ќ	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ђ

Ќtotal

ўcount
Ў
_fn_kwargs
џ	variables
Џ	keras_api"┤
_tf_keras_metricЎ{"class_name": "CategoricalAccuracy", "name": "categorical_accuracy", "dtype": "float32", "config": {"name": "categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
0
Њ0
ћ1"
trackable_list_wrapper
.
Ћ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ќ0
ў1"
trackable_list_wrapper
.
џ	variables"
_generic_user_object
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
-:+└(2 Adam/batch_normalization/gamma/m
,:*└(2Adam/batch_normalization/beta/m
%:#
└(ђ2Adam/dense/kernel/m
:ђ2Adam/dense/bias/m
':%
ђђ2Adam/dense_1/kernel/m
 :ђ2Adam/dense_1/bias/m
&:$	ђ
2Adam/dense_2/kernel/m
:
2Adam/dense_2/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
-:+└(2 Adam/batch_normalization/gamma/v
,:*└(2Adam/batch_normalization/beta/v
%:#
└(ђ2Adam/dense/kernel/v
:ђ2Adam/dense/bias/v
':%
ђђ2Adam/dense_1/kernel/v
 :ђ2Adam/dense_1/bias/v
&:$	ђ
2Adam/dense_2/kernel/v
:
2Adam/dense_2/bias/v
­2ь
 __inference__wrapped_model_25670╚
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0
random_crop_input           
Р2▀
E__inference_sequential_layer_call_and_return_conditional_losses_26836
E__inference_sequential_layer_call_and_return_conditional_losses_26141
E__inference_sequential_layer_call_and_return_conditional_losses_26702
E__inference_sequential_layer_call_and_return_conditional_losses_26257└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ш2з
*__inference_sequential_layer_call_fn_26904
*__inference_sequential_layer_call_fn_26871
*__inference_sequential_layer_call_fn_26524
*__inference_sequential_layer_call_fn_26375└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
&__inference_conv2d_layer_call_fn_26924б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_conv2d_layer_call_and_return_conditional_losses_26915б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv2d_1_layer_call_fn_26944б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26935б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ћ2њ
-__inference_max_pooling2d_layer_call_fn_25682Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
░2Г
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25676Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ї2Ѕ
'__inference_dropout_layer_call_fn_26971
'__inference_dropout_layer_call_fn_26966┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┬2┐
B__inference_dropout_layer_call_and_return_conditional_losses_26956
B__inference_dropout_layer_call_and_return_conditional_losses_26961┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Л2╬
'__inference_flatten_layer_call_fn_26982б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_flatten_layer_call_and_return_conditional_losses_26977б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ц2А
3__inference_batch_normalization_layer_call_fn_27064
3__inference_batch_normalization_layer_call_fn_27051┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌2О
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27018
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27038┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
¤2╠
%__inference_dense_layer_call_fn_27084б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_dense_layer_call_and_return_conditional_losses_27075б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љ2Ї
)__inference_dropout_1_layer_call_fn_27111
)__inference_dropout_1_layer_call_fn_27106┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
к2├
D__inference_dropout_1_layer_call_and_return_conditional_losses_27096
D__inference_dropout_1_layer_call_and_return_conditional_losses_27101┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Л2╬
'__inference_dense_1_layer_call_fn_27131б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_1_layer_call_and_return_conditional_losses_27122б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љ2Ї
)__inference_dropout_2_layer_call_fn_27153
)__inference_dropout_2_layer_call_fn_27158┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
к2├
D__inference_dropout_2_layer_call_and_return_conditional_losses_27143
D__inference_dropout_2_layer_call_and_return_conditional_losses_27148┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Л2╬
'__inference_dense_2_layer_call_fn_27178б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_2_layer_call_and_return_conditional_losses_27169б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
нBЛ
#__inference_signature_wrapper_26565random_crop_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 г
 __inference__wrapped_model_25670Є01/.67@AJKBб?
8б5
3і0
random_crop_input           
ф "1ф.
,
dense_2!і
dense_2         
Х
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27018d01/.4б1
*б'
!і
inputs         └(
p
ф "&б#
і
0         └(
џ Х
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27038d01/.4б1
*б'
!і
inputs         └(
p 
ф "&б#
і
0         └(
џ ј
3__inference_batch_normalization_layer_call_fn_27051W01/.4б1
*б'
!і
inputs         └(
p
ф "і         └(ј
3__inference_batch_normalization_layer_call_fn_27064W01/.4б1
*б'
!і
inputs         └(
p 
ф "і         └(│
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26935l7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         @
џ І
(__inference_conv2d_1_layer_call_fn_26944_7б4
-б*
(і%
inputs          
ф " і         @▒
A__inference_conv2d_layer_call_and_return_conditional_losses_26915l7б4
-б*
(і%
inputs         
ф "-б*
#і 
0          
џ Ѕ
&__inference_conv2d_layer_call_fn_26924_7б4
-б*
(і%
inputs         
ф " і          ц
B__inference_dense_1_layer_call_and_return_conditional_losses_27122^@A0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ |
'__inference_dense_1_layer_call_fn_27131Q@A0б-
&б#
!і
inputs         ђ
ф "і         ђБ
B__inference_dense_2_layer_call_and_return_conditional_losses_27169]JK0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         

џ {
'__inference_dense_2_layer_call_fn_27178PJK0б-
&б#
!і
inputs         ђ
ф "і         
б
@__inference_dense_layer_call_and_return_conditional_losses_27075^670б-
&б#
!і
inputs         └(
ф "&б#
і
0         ђ
џ z
%__inference_dense_layer_call_fn_27084Q670б-
&б#
!і
inputs         └(
ф "і         ђд
D__inference_dropout_1_layer_call_and_return_conditional_losses_27096^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ д
D__inference_dropout_1_layer_call_and_return_conditional_losses_27101^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ~
)__inference_dropout_1_layer_call_fn_27106Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђ~
)__inference_dropout_1_layer_call_fn_27111Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђд
D__inference_dropout_2_layer_call_and_return_conditional_losses_27143^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ д
D__inference_dropout_2_layer_call_and_return_conditional_losses_27148^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ~
)__inference_dropout_2_layer_call_fn_27153Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђ~
)__inference_dropout_2_layer_call_fn_27158Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ▓
B__inference_dropout_layer_call_and_return_conditional_losses_26956l;б8
1б.
(і%
inputs         		@
p
ф "-б*
#і 
0         		@
џ ▓
B__inference_dropout_layer_call_and_return_conditional_losses_26961l;б8
1б.
(і%
inputs         		@
p 
ф "-б*
#і 
0         		@
џ і
'__inference_dropout_layer_call_fn_26966_;б8
1б.
(і%
inputs         		@
p
ф " і         		@і
'__inference_dropout_layer_call_fn_26971_;б8
1б.
(і%
inputs         		@
p 
ф " і         		@Д
B__inference_flatten_layer_call_and_return_conditional_losses_26977a7б4
-б*
(і%
inputs         		@
ф "&б#
і
0         └(
џ 
'__inference_flatten_layer_call_fn_26982T7б4
-б*
(і%
inputs         		@
ф "і         └(в
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25676ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_max_pooling2d_layer_call_fn_25682ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╬
E__inference_sequential_layer_call_and_return_conditional_losses_26141ёY01/.67@AJKJбG
@б=
3і0
random_crop_input           
p

 
ф "%б"
і
0         

џ ═
E__inference_sequential_layer_call_and_return_conditional_losses_26257Ѓ01/.67@AJKJбG
@б=
3і0
random_crop_input           
p 

 
ф "%б"
і
0         

џ ┬
E__inference_sequential_layer_call_and_return_conditional_losses_26702yY01/.67@AJK?б<
5б2
(і%
inputs           
p

 
ф "%б"
і
0         

џ ┴
E__inference_sequential_layer_call_and_return_conditional_losses_26836x01/.67@AJK?б<
5б2
(і%
inputs           
p 

 
ф "%б"
і
0         

џ Ц
*__inference_sequential_layer_call_fn_26375wY01/.67@AJKJбG
@б=
3і0
random_crop_input           
p

 
ф "і         
ц
*__inference_sequential_layer_call_fn_26524v01/.67@AJKJбG
@б=
3і0
random_crop_input           
p 

 
ф "і         
џ
*__inference_sequential_layer_call_fn_26871lY01/.67@AJK?б<
5б2
(і%
inputs           
p

 
ф "і         
Ў
*__inference_sequential_layer_call_fn_26904k01/.67@AJK?б<
5б2
(і%
inputs           
p 

 
ф "і         
─
#__inference_signature_wrapper_26565ю01/.67@AJKWбT
б 
MфJ
H
random_crop_input3і0
random_crop_input           "1ф.
,
dense_2!і
dense_2         
