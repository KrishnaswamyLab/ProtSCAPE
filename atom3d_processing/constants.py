"""

"""


"""
A Python dictionary of atomic weights in Decimal
From:
https://github.com/HegemanLab/atomicWeightsDecimal/blob/master/atomicWeightsDecimal.py
"""

from decimal import Decimal

atomicWeightsDecimal = {
	"H":	{
        "standard":Decimal((0,(1,0,0,7,9,4),-5)),
		"abundant":Decimal((0,(1,0,0,7,8,2,5,0,3,1,9),-10))
		},
	"He":	{
		"standard":Decimal((0,(4,0,0,2,6,0,2),-6)),
		"abundant":Decimal((0,(4,0,0,2,6,0,3,2,4,9,7),-10))
		},
	"Li":	{
		"standard":Decimal((0,(6,9,4,1),-3)),
		"abundant":Decimal((0,(7,0,1,6,0,0,4,1),-7))
		},
	"Be":	{
		"standard":Decimal((0,(9,0,1,2,1,8,2),-6)),
		"abundant":Decimal((0,(9,0,1,2,1,8,2,2),-7))
		},
	"B":	{
		"standard":Decimal((0,(1,0,8,1,1),-3)),
		"abundant":Decimal((0,(1,1,0,0,9,3,0,5,5),-7))
		},
	"C":	{
		"standard":Decimal((0,(1,2,0,1,0,7),-4)),
		"abundant":Decimal((0,(1,2,0,0,0,0,0,0,0,0,0,0),-10))
		},
	"N":	{
		"standard":Decimal((0,(1,4,0,0,6,7),-4)),
		"abundant":Decimal((0,(1,4,0,0,3,0,7,4,0,0,7,4),-10))
                },
	"O":	{
		"standard":Decimal((0,(1,5,9,9,9,4),-4)),
		"abundant":Decimal((0,(1,5,9,9,4,9,1,4,6,2,2,3),-10))
		},
	"F":	{
		"standard":Decimal((0,(1,8,9,9,8,4,0,3,2),-7)),
		"abundant":Decimal((0,(1,8,9,9,8,4,0,3,2,0),-8))
		},
	"Ne":	{
		"standard":Decimal((0,(2,0,1,7,9,7),-5)),
		"abundant":Decimal((0,(1,9,9,9,2,4,4,0,1,7,6),-9))
		},
	"Na":	{
		"standard":Decimal((0,(2,2,9,8,9,7,7,0),-6)),
		"abundant":Decimal((0,(2,2,9,8,9,7,6,9,6,6),-8))
		},
	"Mg":	{
		"standard":Decimal((0,(2,4,3,0,5,0),-4)),
		"abundant":Decimal((0,(2,3,9,8,5,0,4,1,8,7),-8))
		},
	"Al":	{
		"standard":Decimal((0,(2,6,9,8,1,5,3,8),-6)),
		"abundant":Decimal((0,(2,6,9,8,1,5,3,8,4,1),-8))
		},
	"Si":	{
		"standard":Decimal((0,(2,8,0,8,5,5),-4)),
		"abundant":Decimal((0,(2,7,9,7,6,9,2,6,4,9),-8))
		},
	"P":	{
		"standard":Decimal((0,(3,0,9,7,3,7,6,1),-6)),
		"abundant":Decimal((0,(3,0,9,7,3,7,6,1,4,9),-8))
		},
	"S":	{
		"standard":Decimal((0,(3,2,0,6,5),-3)),
		"abundant":Decimal((0,(3,1,9,7,2,0,7,0,7,3),-8))
		},
	"Cl":	{
		"standard":Decimal((0,(3,5,4,5,3),-3)),
		"abundant":Decimal((0,(3,4,9,6,8,8,5,2,7,1),-8))
		},
	"Ar":	{
		"standard":Decimal((0,(3,9,9,4,8),-3)),
		"abundant":Decimal((0,(3,9,9,6,2,3,8,3,1,2,4),-9))
		},
	"K":	{
		"standard":Decimal((0,(3,9,0,9,8,3),-4)),
		"abundant":Decimal((0,(3,8,9,6,3,7,0,6,9),-7))
		},
	"Ca":	{
		"standard":Decimal((0,(4,0,0,7,8),-3)),
		"abundant":Decimal((0,(3,9,9,6,2,5,9,1,2),-7))
		},
	"Sc":	{
		"standard":Decimal((0,(4,4,9,5,5,9,1,0),-6)),
		"abundant":Decimal((0,(4,4,9,5,5,9,1,0,2),-7))
		},
	"Ti":	{
		"standard":Decimal((0,(4,7,8,6,7),-3)),
		"abundant":Decimal((0,(4,7,9,4,7,9,4,7,0),-7))
		},
	"V":	{
		"standard":Decimal((0,(5,0,9,4,1,5),-4)),
		"abundant":Decimal((0,(5,0,9,4,3,9,6,3,5),-7))
		},
	"Cr":	{
		"standard":Decimal((0,(5,1,9,9,6,1),-4)),
		"abundant":Decimal((0,(5,1,9,4,0,5,1,1,5),-7))
		},
	"Mn":	{
		"standard":Decimal((0,(5,4,9,3,8,0,4,9),-6)),
		"abundant":Decimal((0,(5,4,9,3,8,0,4,9,3),-7))
		},
	"Fe":	{
		"standard":Decimal((0,(5,5,8,4,5),-3)),
		"abundant":Decimal((0,(5,5,9,3,4,9,4,1,8),-7))
		},
	"Co":	{
		"standard":Decimal((0,(5,8,9,3,3,2,0,0),-6)),
		"abundant":Decimal((0,(5,8,9,3,3,1,9,9,9),-7))
		},
	"Ni":	{
		"standard":Decimal((0,(5,8,6,9,3,4),-4)),
		"abundant":Decimal((0,(5,7,9,3,5,3,4,7,7),-7))
		},
	"Cu":	{
		"standard":Decimal((0,(6,3,5,4,6),-3)),
		"abundant":Decimal((0,(6,2,9,2,9,6,0,0,7),-7))
		},
	"Zn":	{
		"standard":Decimal((0,(6,5,4,0,9),-3)),
		"abundant":Decimal((0,(6,3,9,2,9,1,4,6,1),-7))
		},
	"Ga":	{
		"standard":Decimal((0,(6,9,7,2,3),-3)),
		"abundant":Decimal((0,(6,8,9,2,5,5,8,1),-6))
		},
	"Ge":	{
		"standard":Decimal((0,(7,2,6,4),-2)),
		"abundant":Decimal((0,(7,3,9,2,1,1,7,8,4),-7))
		},
	"As":	{
		"standard":Decimal((0,(7,4,9,2,1,6,0),-5)),
		"abundant":Decimal((0,(7,4,9,2,1,5,9,6,6),-7))
		},
	"Se":	{
		"standard":Decimal((0,(7,8,9,6),-2)),
		"abundant":Decimal((0,(7,7,9,1,6,5,2,2,1),-7))
		},
	"Br":	{
		"standard":Decimal((0,(7,9,9,0,4),-3)),
		"abundant":Decimal((0,(7,8,9,1,8,3,3,7,9),-7))
		},
	"Kr":	{
		"standard":Decimal((0,(8,3,7,9,8),-3)),
		"abundant":Decimal((0,(8,3,9,1,1,5,0,8),-6))
		},
	"Rb":	{
		"standard":Decimal((0,(8,5,4,6,7,8),-4)),
		"abundant":Decimal((0,(8,4,9,1,1,7,9,2,4),-7))
		},
	"Sr":	{
		"standard":Decimal((0,(8,7,6,2),-2)),
		"abundant":Decimal((0,(8,7,9,0,5,6,1,6,7),-7))
		},
	"Y":	{
		"standard":Decimal((0,(8,8,9,0,5,8,5),-5)),
		"abundant":Decimal((0,(8,8,9,0,5,8,4,8,5),-7))
		},
	"Zr":	{
		"standard":Decimal((0,(9,1,2,2,4),-3)),
		"abundant":Decimal((0,(8,9,9,0,4,7,0,2,2),-7))
		},
	"Nb":	{
		"standard":Decimal((0,(9,2,9,0,6,3,8),-5)),
		"abundant":Decimal((0,(9,2,9,0,6,3,7,6,2),-7))
		},
	"Mo":	{
		"standard":Decimal((0,(9,5,9,4),-2)),
		"abundant":Decimal((0,(9,7,9,0,5,4,0,6,9),-7))
		},
	"Ru":	{
		"standard":Decimal((0,(1,0,1,0,7),-2)),
		"abundant":Decimal((0,(1,0,1,9,0,4,3,4,8,8),-7))
		},
	"Rh":	{
		"standard":Decimal((0,(1,0,2,9,0,5,5,0),-5)),
		"abundant":Decimal((0,(1,0,2,9,0,5,5,0,4),-6))
		},
	"Pd":	{
		"standard":Decimal((0,(1,0,6,4,2),-2)),
		"abundant":Decimal((0,(1,0,5,9,0,3,4,8,4),-6))
		},
	"Ag":	{
		"standard":Decimal((0,(1,0,7,8,6,8,2),-4)),
		"abundant":Decimal((0,(1,0,6,9,0,5,0,9,3),-6))
		},
	"Cd":	{
		"standard":Decimal((0,(1,1,2,4,1,1),-3)),
		"abundant":Decimal((0,(1,1,3,9,0,3,3,5,8,6),-7))
		},
	"In":	{
		"standard":Decimal((0,(1,1,4,8,1,8),-3)),
		"abundant":Decimal((0,(1,1,4,9,0,3,8,7,9),-6))
		},
	"Sn":	{
		"standard":Decimal((0,(1,1,8,7,1,0),-3)),
		"abundant":Decimal((0,(1,1,9,9,0,2,1,9,8,5),-7))
		},
	"Sb":	{
		"standard":Decimal((0,(1,2,1,7,6,0),-3)),
		"abundant":Decimal((0,(1,2,0,9,0,3,8,2,2,2),-7))
		},
	"Te":	{
		"standard":Decimal((0,(1,2,7,6,0),-2)),
		"abundant":Decimal((0,(1,2,9,9,0,6,2,2,2,9),-7))
		},
	"I":	{
		"standard":Decimal((0,(1,2,6,9,0,4,4,7),-5)),
		"abundant":Decimal((0,(1,2,6,9,0,4,4,6,8),-6))
		},
	"Xe":	{
		"standard":Decimal((0,(1,3,1,2,9,3),-3)),
		"abundant":Decimal((0,(1,3,1,9,0,4,1,5,4,6),-7))
		},
	"Cs":	{
		"standard":Decimal((0,(1,3,2,9,0,5,4,5),-5)),
		"abundant":Decimal((0,(1,3,2,9,0,5,4,4,7),-6))
		},
	"Ba":	{
		"standard":Decimal((0,(1,3,7,3,2,7),-3)),
		"abundant":Decimal((0,(1,3,7,9,0,5,2,4,2),-6))
		},
	"La":	{
		"standard":Decimal((0,(1,3,8,9,0,5,5),-4)),
		"abundant":Decimal((0,(1,3,8,9,0,6,3,4,9),-6))
		},
	"Ce":	{
		"standard":Decimal((0,(1,4,0,1,1,6),-3)),
		"abundant":Decimal((0,(1,3,9,9,0,5,4,3,5),-6))
		},
	"Pr":	{
		"standard":Decimal((0,(1,4,0,9,0,7,6,5),-5)),
		"abundant":Decimal((0,(1,4,0,9,0,7,6,4,8),-6))
		},
	"Nd":	{
		"standard":Decimal((0,(1,4,4,2,4),-2)),
		"abundant":Decimal((0,(1,4,1,9,0,7,7,1,9),-6))
		},
	"Sm":	{
		"standard":Decimal((0,(1,5,0,3,6),-2)),
		"abundant":Decimal((0,(1,5,1,9,1,9,7,2,9),-6))
		},
	"Eu":	{
		"standard":Decimal((0,(1,5,1,9,6,4),-3)),
		"abundant":Decimal((0,(1,5,2,9,2,1,2,2,7),-6))
		},
	"Gd":	{
		"standard":Decimal((0,(1,5,7,2,5),-2)),
		"abundant":Decimal((0,(1,5,7,9,2,4,1,0,1),-6))
		},
	"Tb":	{
		"standard":Decimal((0,(1,5,8,9,2,5,3,4),-5)),
		"abundant":Decimal((0,(1,5,8,9,2,5,3,4,3),-6))
		},
	"Dy":	{
		"standard":Decimal((0,(1,6,2,5,0,0),-3)),
		"abundant":Decimal((0,(1,6,3,9,2,9,1,7,1),-6))
		},
	"Ho":	{
		"standard":Decimal((0,(1,6,4,9,3,0,3,2),-5)),
		"abundant":Decimal((0,(1,6,4,9,3,0,3,1,9),-6))
		},
	"Er":	{
		"standard":Decimal((0,(1,6,7,2,5,9),-3)),
		"abundant":Decimal((0,(1,6,5,9,3,0,2,9,0),-6))
		},
	"Tm":	{
		"standard":Decimal((0,(1,6,8,9,3,4,2,1),-5)),
		"abundant":Decimal((0,(1,6,8,9,3,4,2,1,1),-6))
		},
	"Yb":	{
		"standard":Decimal((0,(1,7,3,0,4),-2)),
		"abundant":Decimal((0,(1,7,3,9,3,8,8,5,8),-6))
		},
	"Lu":	{
		"standard":Decimal((0,(1,7,4,9,6,7),-3)),
		"abundant":Decimal((0,(1,7,4,9,4,0,7,6,8,2),-7))
		},
	"Hf":	{
		"standard":Decimal((0,(1,7,8,4,9),-2)),
		"abundant":Decimal((0,(1,7,9,9,4,6,5,4,8,8),-7))
		},
	"Ta":	{
		"standard":Decimal((0,(1,8,0,9,4,7,9),-4)),
		"abundant":Decimal((0,(1,8,0,9,4,7,9,9,6),-6))
		},
	"W":	{
		"standard":Decimal((0,(1,8,3,8,4),-2)),
		"abundant":Decimal((0,(1,8,3,9,5,0,9,3,2,3),-7))
		},
	"Re":	{
		"standard":Decimal((0,(1,8,6,2,0,7),-3)),
		"abundant":Decimal((0,(1,8,6,9,5,5,7,5,0,5),-7))
		},
	"Os":	{
		"standard":Decimal((0,(1,9,0,2,3),-2)),
		"abundant":Decimal((0,(1,9,1,9,6,1,4,7,9),-6))
		},
	"Ir":	{
		"standard":Decimal((0,(1,9,2,2,1,7),-3)),
		"abundant":Decimal((0,(1,9,2,9,6,2,9,2,3),-6))
		},
	"Pt":	{
		"standard":Decimal((0,(1,9,5,0,7,8),-3)),
		"abundant":Decimal((0,(1,9,4,9,6,4,7,7,4),-6))
		},
	"Au":	{
		"standard":Decimal((0,(1,9,6,9,6,6,5,5),-5)),
		"abundant":Decimal((0,(1,9,6,9,6,6,5,5,1),-6))
		},
	"Hg":	{
		"standard":Decimal((0,(2,0,0,5,9),-2)),
		"abundant":Decimal((0,(2,0,1,9,7,0,6,2,5),-6))
		},
	"Tl":	{
		"standard":Decimal((0,(2,0,4,3,8,3,3),-4)),
		"abundant":Decimal((0,(2,0,4,9,7,4,4,1,2),-6))
		},
	"Pb":	{
		"standard":Decimal((0,(2,0,7,2),-1)),
		"abundant":Decimal((0,(2,0,7,9,7,6,6,3,6),-6))
		},
	"Bi":	{
		"standard":Decimal((0,(2,0,8,9,8,0,3,8),-5)),
		"abundant":Decimal((0,(2,0,8,9,8,0,3,8,4),-6))
		},
	"Th":	{
		"standard":Decimal((0,(2,3,2,0,3,8,1),-4)),
		"abundant":Decimal((0,(2,3,2,0,3,8,0,4,9,5),-7))
		},
	"Pa":	{
		"standard":Decimal((0,(2,3,1,0,3,5,8,8),-5)),
		"abundant":Decimal((0,(2,3,1,0,3,5,8,8),-5))
		},
	"U":	{
		"standard":Decimal((0,(2,3,8,0,2,8,9,1),-5)),
		"abundant":Decimal((0,(2,3,8,0,5,0,7,8,3,5),-7))
                }
}