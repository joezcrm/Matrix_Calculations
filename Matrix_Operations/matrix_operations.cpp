#include "matrix_operations.h"
#include <iomanip>
#include <thread>
#include <future>
#include <chrono>
#include <cmath>
#include <algorithm>

const MInt threadLimit = 32;

void Householder::initialize(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt vLocation, MInt vSize,
	MInt nStart, bool useCol)
{
	// Should remove the error checking, add here for future modification
	if (colSize == 0 || rowSize == 0 || vSize == 0)
	{
		throw std::invalid_argument("At least one index is improper.");
	}
	if (!dMatrix)
	{
		throw std::invalid_argument("Matrix cannot be nullptr.");
	}
	h_Size = vSize;
	h_Start = nStart;
	h_Vector = new LDouble[h_Size];
	if (!h_Vector)
	{
		throw std::bad_alloc();
	}
	h_Beta = 0.0L;
	// Divide the the vector into several intervals and sum up each intervals
	// To reduce numeric error
	MInt intervalNum = h_Size / VECTOR_INTERVAL_SIZE;
	intervalNum++;
	MInt intervalLimit = h_Size / intervalNum;
	intervalLimit++;
	MInt counter;
	LDouble dSum;
	LDouble maxDouble;
	// Use a column to generate Householder
	if (useCol)
	{
		// Error checking should remove
		if (vLocation >= rowSize || nStart + h_Size > colSize)
		{
			throw std::invalid_argument("Location can not be greater than size.");
		}
		h_Dim = colSize;
		maxDouble = abs(dMatrix[vLocation][nStart]);
		for (MInt i = nStart + 1; i < nStart + h_Size; i++)
		{
			if (maxDouble < abs(dMatrix[vLocation][i]))
			{
				maxDouble = abs(dMatrix[vLocation][i]);
			}
		}
		h_Alpha = 0.0L;
		if (maxDouble < NUMERIC_ERROR)
		{
			for (MInt i = 0; i < h_Size; i++)
			{
				h_Vector[i] = 0.0L;
			}
		}
		else
		{
			// Sum up
			for (MInt i = 0; i < intervalNum - 1; i++)
			{
				dSum = 0.0L;
				for (counter = 0; counter < intervalLimit; counter++)
				{
					h_Vector[i * intervalLimit + counter] =
						dMatrix[vLocation][nStart + i * intervalLimit + counter] / maxDouble;
					dSum += (h_Vector[i * intervalLimit + counter] 
						* h_Vector[i * intervalLimit + counter]);
				}
				h_Alpha += dSum;
			}
			dSum = 0.0L;
			for (counter = (intervalNum - 1) * intervalLimit; counter < h_Size; counter++)
			{
				h_Vector[counter] = dMatrix[vLocation][nStart + counter] / maxDouble;
				dSum += (h_Vector[counter] * h_Vector[counter]);
			}
			h_Alpha += dSum;
		}
	}
	else
	{
		// Use a row to generate Householder
		// Remove error checking
		if (vLocation >= colSize || nStart + h_Size > rowSize)
		{
			throw std::invalid_argument("Location can not be greater than size.");
		}
		h_Dim = rowSize;
		maxDouble = abs(dMatrix[nStart][vLocation]);
		for (MInt i = nStart + 1; i < nStart + h_Size; i++)
		{
			if (maxDouble < abs(dMatrix[i][vLocation]))
			{
				maxDouble = abs(dMatrix[i][vLocation]);
			}
		}
		h_Alpha = 0.0L;
		if (maxDouble < NUMERIC_ERROR)
		{
			for (MInt i = 0; i < h_Size; i++)
			{
				h_Vector[i] = 0.0L;
			}
		}
		else
		{
			// Sum up
			for (MInt i = 0; i < intervalNum - 1; i++)
			{
				dSum = 0.0L;
				for (counter = 0; counter < intervalLimit; counter++)
				{
					h_Vector[i * intervalLimit + counter] =
						dMatrix[nStart + i * intervalLimit + counter][vLocation] / maxDouble;
					dSum += (h_Vector[i * intervalLimit + counter]
						* h_Vector[i * intervalLimit + counter]);
				}
				h_Alpha += dSum;
			}
			dSum = 0.0L;
			for (counter = (intervalNum - 1) * intervalLimit; counter < h_Size; counter++)
			{
				h_Vector[counter] = dMatrix[nStart + counter][vLocation] / maxDouble;
				dSum += (h_Vector[counter] * h_Vector[counter]);
			}
			h_Alpha += dSum;
		}
	}
	h_Alpha = sqrtl(h_Alpha);
	if (h_Vector[0] < 0.0L)
	{
		h_Alpha *= (-1.0L);
	}
	h_Vector[0] += h_Alpha;
	if (maxDouble >= NUMERIC_ERROR)
	{
		h_Beta = 1.0L / (h_Vector[0] * h_Alpha);
	}
	h_Alpha *= maxDouble;
}

Householder::Householder(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt vLocation, MInt vSize, bool useCol)
{
	if (useCol)
	{
		// Use column
		initialize(dMatrix, colSize, rowSize, vLocation, vSize, colSize - vSize, true);
	}
	else
	{
		// use row
		initialize(dMatrix, colSize, rowSize, vLocation, vSize, rowSize - vSize, false);
	}
}
Householder::Householder(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt vLocation, MInt vSize,
	MInt nStart, bool useCol)
{
	initialize(dMatrix, colSize, rowSize, vLocation, vSize, nStart, useCol);
}

// Should not use default Householder directly
Householder::Householder()
{
	h_Dim = 1;
	h_Beta = 0.0L;
	h_Alpha = 0.0L;
	h_Size = 1;
	h_Start = 1;
	h_Vector = nullptr;
}
Householder::~Householder()
{
	if (h_Vector)
	{
		delete[] h_Vector;
	}
}

Householder& Householder::operator =(Householder& hOther)
{
	h_Dim = hOther.h_Dim;
	h_Beta = hOther.h_Beta;
	h_Alpha = hOther.h_Alpha;
	h_Size = hOther.h_Size;
	h_Start = hOther.h_Start;
	if (h_Vector)
	{
		delete[] h_Vector;
	}
	h_Vector = new LDouble[h_Size];
	for (MInt i = 0; i < h_Size; i++)
	{
		h_Vector[i] = hOther.h_Vector[i];
	}
	return *this;
}

// Multiply a column
void Householder::leftMultiply(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt colLocation) const
{
	// Remove error checking
	if (colSize != h_Dim || rowSize == 0 || colLocation >= rowSize)
	{
		throw std::invalid_argument("At least one index was incorrect.");
	}
	if (!dMatrix)
	{
		throw std::invalid_argument("Matrix doesn't contains any values.");
	}
	// Divide vector into several intervals
	MInt intervalNum = h_Size / VECTOR_INTERVAL_SIZE;
	intervalNum++;
	MInt intervalLimit = h_Size / intervalNum;
	intervalLimit++;
	MInt counter;
	LDouble innerProduct = 0.0L;
	LDouble dSum;
	// Calculate inner product
	for (MInt i = 0; i < (intervalNum - 1); i++)
	{
		dSum = 0.0L;
		for (counter = 0; counter < intervalLimit; counter++)
		{
			dSum += (h_Vector[i * intervalLimit + counter]
				* dMatrix[colLocation][h_Start + i * intervalLimit + counter]);
		}
		innerProduct += dSum;
	}
	dSum = 0.0L;
	for (counter = (intervalNum - 1) * intervalLimit; counter < h_Size; counter++)
	{
		dSum += (h_Vector[counter] * dMatrix[colLocation][h_Start + counter]);
	}
	innerProduct += dSum;
	// Calculate the resulting vector 
	for (MInt i = 0; i < h_Size; i++)
	{
		dMatrix[colLocation][h_Start + i] -= (h_Beta * innerProduct * h_Vector[i]);
	}
}

// Multiply a row
void Householder::rightMultiply(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt rowLocation) const
{
	// Remove error checking
	if (rowSize != h_Dim || colSize == 0 || rowLocation >= colSize)
	{
		throw std::invalid_argument("At least one argument is invalid.");
	}
	// Divide vector into intervals
	MInt intervalNum = h_Size / VECTOR_INTERVAL_SIZE;
	intervalNum++;
	MInt intervalLimit = h_Size / intervalNum;
	intervalLimit++;
	MInt counter;
	LDouble innerProduct = 0.0L;
	LDouble dSum;
	// Calculate inner product
	for (MInt i = 0; i < (intervalNum - 1); i++)
	{
		dSum = 0.0L;
		for (counter = 0; counter < intervalLimit; counter++)
		{
			dSum += (h_Vector[i * intervalLimit + counter]
				* dMatrix[h_Start + i * intervalLimit + counter][rowLocation]);
		}
		innerProduct += dSum;
	}
	dSum = 0.0L;
	for (counter = (intervalNum - 1) * intervalLimit; counter < h_Size; counter++)
	{
		dSum += (h_Vector[counter] * dMatrix[h_Start + counter][rowLocation]);
	}
	innerProduct += dSum;
	// Calculate resulting vector
	for (MInt i = 0; i < h_Size; i++)
	{
		dMatrix[h_Start + i][rowLocation] -= (h_Beta * innerProduct * h_Vector[i]);
	}
}

LDouble Householder::getAlpha() { return h_Alpha; }

Givens::Givens(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt vLocation,
	MInt locationK, MInt locationI, bool useCol)
{
	// Remove error checking
	if (colSize == 0 || rowSize == 0 || locationK == locationI) 
	{
		throw std::invalid_argument("Size cannot be zero.");
	}
	if (!dMatrix)
	{
		throw std::invalid_argument("Matrix cannot be nullptr.");
	}
	g_LocationK = locationK;
	g_LocationI = locationI;
	if (useCol)
	{
		// Use a column from a matrix
		// Remove error checking
		if (vLocation >= rowSize)
		{
			throw std::invalid_argument("Location of vector cannot be greater than row size.");
		}
		if (g_LocationK >= colSize || g_LocationI >= colSize)
		{
			throw std::invalid_argument("Location with a vector cannot be greater than size.");
		}
		g_Dim = colSize;
		if (abs(dMatrix[vLocation][g_LocationK]) < NUMERIC_ERROR * NUMERIC_ERROR)
		{
			g_Cos = 1.0L;
			g_Sin = 0.0L;
		}
		else
		{
			// Calculate resulting vector
			LDouble quotient;
			if (abs(dMatrix[vLocation][g_LocationK]) >= abs(dMatrix[vLocation][g_LocationI]))
			{
				quotient = dMatrix[vLocation][g_LocationI] / dMatrix[vLocation][g_LocationK];
				g_Sin = 1.0L / (sqrtl(1.0L + quotient * quotient));
				g_Cos = g_Sin * quotient;
			}
			else
			{
				quotient = dMatrix[vLocation][g_LocationK] / dMatrix[vLocation][g_LocationI];
				g_Cos = 1.0L / (sqrtl(1.0L + quotient * quotient));
				g_Sin = g_Cos * quotient;
			}
		}
	}
	else
	{
		// Use a row from matrix
		// Remove error checking
		if (vLocation >= colSize)
		{
			throw std::invalid_argument("Location of vector cannot be greater than column size.");
		}
		if (g_LocationK >= rowSize || g_LocationI >= rowSize)
		{
			throw std::invalid_argument("Location within a vector cannot be greater than size.");
		}
		g_Dim = rowSize;
		if (abs(dMatrix[g_LocationK][vLocation]) < NUMERIC_ERROR * NUMERIC_ERROR)
		{
			g_Cos = 1.0L;
			g_Sin = 0.0L;
		}
		else
		{
			// Calculate resulting vector
			LDouble quotient;
			if (abs(dMatrix[g_LocationK][vLocation]) >= abs(dMatrix[g_LocationI][vLocation]))
			{
				quotient = dMatrix[g_LocationI][vLocation] / dMatrix[g_LocationK][vLocation];
				g_Sin = 1.0L / (sqrtl(1.0L + quotient * quotient));
				g_Cos = g_Sin * quotient;
			}
			else
			{
				quotient = dMatrix[g_LocationK][vLocation] / dMatrix[g_LocationI][vLocation];
				g_Cos = 1.0L / sqrtl(1.0L + quotient * quotient);
				g_Sin = g_Cos * quotient;
			}
		}
	}
}

// Should not use default Givens matrix directly
Givens::Givens()
{
	g_Dim = 0;
	g_LocationK = 0;
	g_LocationI = 0;
	g_Cos = 0.0L;
	g_Sin = 0.0L;
}

Givens::~Givens() {}

Givens& Givens::operator = (Givens& gOther)
{
	g_Dim = gOther.g_Dim;
	g_Cos = gOther.g_Cos;
	g_Sin = gOther.g_Sin;
	g_LocationK = gOther.g_LocationK;
	g_LocationI = gOther.g_LocationI;
	return *this;
}

// Multiply column from matrix
void Givens::leftMultiply(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt vLocation) const
{
	// Remove error checking
	if (g_Dim != colSize)
	{
		throw std::invalid_argument("Column size mismatched.");
	}
	if (rowSize == 0 || vLocation >= rowSize)
	{
		throw std::invalid_argument(
			"Row size cannot be zero and location cannot be greater than row size.");
	}
	LDouble ldK = dMatrix[vLocation][g_LocationK];
	LDouble ldI = dMatrix[vLocation][g_LocationI];
	dMatrix[vLocation][g_LocationK] = (-1.0L) * g_Sin * ldI + g_Cos * ldK;
	dMatrix[vLocation][g_LocationI] = g_Cos * ldI + g_Sin * ldK;
}

// Multiply row from matrix
void Givens::rightMultiply(LDPtr* dMatrix, MInt colSize, MInt rowSize, MInt vLocation) const
{
	// Remove error checking
	if (g_Dim != rowSize)
	{
		throw std::invalid_argument("Row size mismatched.");
	}
	if (colSize == 0 || vLocation >= colSize)
	{
		throw std::invalid_argument(
			"Column size cannot be zero and location cannot be greater than column size.");
	}
	LDouble ldK = dMatrix[g_LocationK][vLocation];
	LDouble ldI = dMatrix[g_LocationI][vLocation];
	dMatrix[g_LocationK][vLocation] = (-1.0L) * g_Sin * ldI + g_Cos * ldK;
	dMatrix[g_LocationI][vLocation] = g_Cos * ldI + g_Sin * ldK;
}

// Intitialize a matrix, pMatrix can be nullptr
void DenseMatrix::initialize(LDPtr* pMatrix, MInt nSize)
{
	if (nSize == 0)
	{
		throw std::invalid_argument("Size cannot be zero.");
	}
	m_Size = nSize;
	m_Thread = NUMBER_OF_THREADING;
	m_Symmetrical = false;
	bool memoryError = false;
	m_TransMatrix = nullptr;
	m_Matrix = nullptr;
	// Set transformation matrix to the identity matrix
	m_TransMatrix = new LDPtr[m_Size];
	if (!m_TransMatrix)
	{
		memoryError = true;
		goto CleanUpMemory;
	}
	m_Matrix = new LDPtr[m_Size];
	if (!m_Matrix)
	{
		memoryError = true;
		goto CleanUpMemory;
	}
	for (MInt i = 0; i < m_Size; i++)
	{
		m_TransMatrix[i] = nullptr;
		m_Matrix[i] = nullptr;
	}
	for (MInt i = 0; i < m_Size; i++)
	{
		m_TransMatrix[i] = new LDouble[m_Size];
		if (!m_TransMatrix[i])
		{
			memoryError = true;
			goto CleanUpMemory;
		}
		m_Matrix[i] = new LDouble[m_Size];
		if (!m_Matrix[i])
		{
			memoryError = true;
			goto CleanUpMemory;
		}
		if (!pMatrix || !pMatrix[i])
		{
			for (MInt j = 0; j < i; j++)
			{
				m_TransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = 0.0L;
			}
			m_TransMatrix[i][i] = 1.0L;
			m_Matrix[i][i] = 0.0L;
			for (MInt j = i + 1; j < m_Size; j++)
			{
				m_TransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = 0.0L;
			}
		}
		else
		{
			for (MInt j = 0; j < i; j++)
			{
				m_TransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = pMatrix[i][j];
			}
			m_TransMatrix[i][i] = 1.0L;
			m_Matrix[i][i] = pMatrix[i][i];
			for (MInt j = i + 1; j < m_Size; j++)
			{
				m_TransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = pMatrix[i][j];
			}
		}
	}
CleanUpMemory:
	// If initialization fails, clear up all memory
	if (memoryError)
	{
		clearMemory();
		throw std::bad_alloc();
	}
}

DenseMatrix::DenseMatrix(LDPtr* pMatrix, MInt nSize)
{
	initialize(pMatrix, nSize);
}

void DenseMatrix::clearMemory()
{
	if (m_TransMatrix)
	{
		for (MInt i = 0; i < m_Size; i++)
		{
			if (m_TransMatrix[i])
			{
				delete[] m_TransMatrix[i];
				m_TransMatrix[i] = nullptr;
			}
		}
		delete[] m_TransMatrix;
		m_TransMatrix = nullptr;
	}
	if (m_Matrix)
	{
		for (MInt i = 0; i < m_Size; i++)
		{
			if (m_Matrix[i])
			{
				delete[] m_Matrix[i];
				m_Matrix[i] = nullptr;
			}
		}
		delete[] m_Matrix;
		m_Matrix = nullptr;
	}
}

DenseMatrix::~DenseMatrix()
{
	clearMemory();
}

void DenseMatrix::setSymmetrical(bool bSymmetrical)
{
	m_Symmetrical = bSymmetrical;
}

// The function is used in multithreading, left multiply by transformation matrix
void leftMultiplyByTransMatrix(std::promise<TDataPtr>* mainPromise,
	std::promise<void>* threadPromise, MInt tSize, MInt tLocation)
{
	// Remove error checking
	if (tSize == 0 || tLocation >= tSize)
	{
		throw std::invalid_argument("At least one argument is improper.");
	}
	// Wait for main thread event
	std::future<TDataPtr> mainFuture = mainPromise[tLocation].get_future();
	TDataPtr DPtr = mainFuture.get();
	// If pointers are nullptr, thread will terminate
	while (DPtr->dMatrix && ((DPtr->byHouse && DPtr->hPtr) || (!DPtr->byHouse && DPtr->gPtr)))
	{
		// If byHouse is true, multiply by Householder
		if (DPtr->byHouse)
		{
			for (MInt i = DPtr->minLocation; i < DPtr->maxLocation; i++)
			{
				DPtr->hPtr->leftMultiply(DPtr->dMatrix, DPtr->colSize, DPtr->rowSize, i);
			}
		}
		else
		{
			// If byHouse is false, multtiply by Givens
			for (MInt i = DPtr->minLocation; i < DPtr->maxLocation; i++)
			{
				DPtr->gPtr->leftMultiply(DPtr->dMatrix, DPtr->colSize, DPtr->rowSize, i);
			}
		}
		// Reset mainPromise status
		std::promise<TDataPtr> temp;
		mainPromise[tLocation] = std::move(temp);
		mainFuture = mainPromise[tLocation].get_future();
		// Notify main thread
		threadPromise[tLocation].set_value();
		// Wait form main thread notification
		DPtr = mainFuture.get();
	}
}

void rightMultiplyByTransMatrix(std::promise<TDataPtr>* mainPromise,
	std::promise<void>* threadPromise, MInt tSize, MInt tLocation)
{
	// Remove error checking
	if (tSize == 0 || tLocation >= tSize)
	{
		throw std::invalid_argument("Sizes are not proper.");
	}
	// Wait from main thread notification
	std::future<TDataPtr> mainFuture = mainPromise[tLocation].get_future();
	TDataPtr DPtr = mainFuture.get();
	// If pointers are nullptr, thread terminates
	while (DPtr->dMatrix && ((DPtr->byHouse && DPtr->hPtr) || (!DPtr->byHouse && DPtr->gPtr)))
	{
		// If byHouse is true, multiply by Householder
		if (DPtr->byHouse)
		{
			for (MInt i = DPtr->minLocation; i < DPtr->maxLocation; i++)
			{
				DPtr->hPtr->rightMultiply(DPtr->dMatrix, DPtr->colSize, DPtr->rowSize, i);
			}
		}
		else
		{
			// If byHouse is false, multiply by Givens
			for (MInt i = DPtr->minLocation; i < DPtr->maxLocation; i++)
			{
				DPtr->gPtr->rightMultiply(DPtr->dMatrix, DPtr->colSize, DPtr->rowSize, i);
			}
		}
		// Reset mainPromise status
		std::promise<TDataPtr> temp;
		mainPromise[tLocation] = std::move(temp);
		mainFuture = std::move(mainPromise[tLocation].get_future());
		// Notify main thread
		threadPromise[tLocation].set_value();
		// Wait for main thread notification
		DPtr = mainFuture.get();
	}
}

void DenseMatrix::setValue(MInt colPos, MInt rowPos, LDouble dValue)
{
	if (colPos >= m_Size || rowPos >= m_Size)
	{
		throw std::invalid_argument("Position cannot be larger than size.");
	}
	m_Matrix[colPos][rowPos] = dValue;
}

LDouble DenseMatrix::getValue(MInt colPos, MInt rowPos)
{
	if (colPos >= m_Size || rowPos >= m_Size)
	{
		throw std::invalid_argument("Position cannot be larger than size.");
	}
	return m_Matrix[colPos][rowPos];
}

// May use this method to solve system of linear equations
// Will solve the system internally
// This method is added here for testing only
void DenseMatrix::toLowerTriangleByHouseholderAndSolve(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Pointer to constant vector cannot be nullptr.");
	}
	if (nSize != m_Size)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	HousePtr pHouse = new Householder[m_Size - 1];
	for (MInt i = m_Size; i > 1; i--)
	{
		Householder hMatrix(m_Matrix, m_Size, m_Size, m_Size - i, i, false);
		pHouse[m_Size - i] = hMatrix;
		for (MInt j = m_Size - i; j < m_Size; j++)
		{
			hMatrix.rightMultiply(m_Matrix, m_Size, m_Size, j);
		}
	}
	solveLowerTriangle(pVector, nSize);
	LDPtr* vectorMatrix = new LDPtr[1];
	vectorMatrix[0] = pVector;
	for (int i = (int)(m_Size)-2; i >= 0; i--)
	{
		pHouse[i].leftMultiply(vectorMatrix, m_Size, 1, 0);
	}
	delete[] vectorMatrix;
	delete[] pHouse;
}

// Transform matrix to upper Hessenberg form by multiply proper Householder on 
//		left and right
void DenseMatrix::toUpperHessenberg()
{
	// Multithreading
	std::promise<TDataPtr>* leftMainPromise = nullptr;
	std::promise<void>* leftThreadPromise = nullptr;
	std::thread* leftThread = nullptr;
	std::promise<TDataPtr>* rightMainPromise = nullptr;
	std::promise<void>* rightThreadPromise = nullptr;
	std::thread* rightThread = nullptr;
	HousePtr pHouse;
	if (m_Thread > 0)
	{
		leftMainPromise = new std::promise<TDataPtr>[m_Thread];
		leftThreadPromise = new std::promise<void>[m_Thread];
		leftThread = new std::thread[m_Thread];
		rightMainPromise = new std::promise<TDataPtr>[m_Thread];
		rightThreadPromise = new std::promise<void>[m_Thread];
		rightThread = new std::thread[m_Thread];
		for (MInt i = 0; i < m_Thread; i++)
		{
			leftThread[i] = std::thread(leftMultiplyByTransMatrix, leftMainPromise,
				leftThreadPromise, m_Thread, i);
			rightThread[i] = std::thread(rightMultiplyByTransMatrix, rightMainPromise,
				rightThreadPromise, m_Thread, i);
		}
	}
	if (m_Size <= 2)
	{
		return;
	}
	// If matrix is symmetrical, certain rows do not need to be multiplied
	if (m_Symmetrical)
	{
		for (MInt i = 0; i < m_Size - 2; i++)
		{
			pHouse = new Householder(m_Matrix, m_Size, m_Size, i, m_Size - i - 1, true);
			// Matrix multiplication
			multiplyByHouseholder(
				leftMainPromise,
				leftThreadPromise,
				rightMainPromise,
				rightThreadPromise,
				pHouse,
				i + 1,
				m_Size,
				i + 1,
				m_Size,
				true);
			// Change elements on the ith column and row
			m_Matrix[i][i + 1] = (-1.0L) * pHouse->getAlpha();
			m_Matrix[i + 1][i] = m_Matrix[i][i + 1];
			for (MInt j = i + 2; j < m_Size; j++)
			{
				m_Matrix[i][j] = 0.0L;
				m_Matrix[j][i] = 0.0L;
			}
			delete pHouse;
		}
	}
	else
	{
		// If not symmetrical, need to multiply every row
		for (MInt i = 0; i < m_Size - 2; i++)
		{
			pHouse = new Householder(m_Matrix, m_Size, m_Size, i, m_Size - i - 1, true);
			multiplyByHouseholder(
				leftMainPromise,
				leftThreadPromise,
				rightMainPromise,
				rightThreadPromise,
				pHouse,
				i + 1,
				m_Size,
				(MInt)0,
				m_Size,
				true);
			// Change elements on ith column only
			m_Matrix[i][i + 1] = (-1.0L) * pHouse->getAlpha();
			for (MInt j = i + 2; j < m_Size; j++)
			{
				m_Matrix[i][j] = 0.0L;
			}
			delete pHouse;
		}
	}
	if (m_Thread > 0)
	{
		ThreadingDatta data;
		data.dMatrix = nullptr;
		data.byHouse = true;
		data.gPtr = nullptr;
		data.hPtr = nullptr;
		for (MInt j = 0; j < m_Thread; j++)
		{
			leftMainPromise[j].set_value(&(data));
			leftThread[j].join();
			rightMainPromise[j].set_value(&(data));
			rightThread[j].join();
		}
		if (leftMainPromise)
		{
			delete[] leftMainPromise;
		}
		if (leftThreadPromise)
		{
			delete[] leftThreadPromise;
		}
		if (leftThread)
		{
			delete[] leftThread;
		}
		if (rightMainPromise)
		{
			delete[] rightMainPromise;
		}
		if (rightThreadPromise)
		{
			delete[] rightThreadPromise;
		}
		if (rightThread)
		{
			delete[] rightThread;
		}
	}
}

// Perform matrix multiplication on left and right
// If transformation matrix is needed, will calculate the transformation matrix
void DenseMatrix::multiplyByHouseholder(
	std::promise<TDataPtr>* leftMainPromise,
	std::promise<void>* leftThreadPromise,
	std::promise<TDataPtr>* rightMainPromise,
	std::promise<void>* rightThreadPromise,
	HousePtr pHMatrix, 
	MInt leftStart,
	MInt leftEnd,
	MInt rightStart,
	MInt rightEnd,
	bool calcTrans)
{
	MInt j;
	MInt minPos;
	MInt nRuns;
	MInt transThread;
	MInt matrixThread;
	if (m_Thread == 0)
	{
		// Perform all calculations in the main thread
		for (j = rightStart; j < rightEnd; j++)
		{
			pHMatrix->rightMultiply(m_Matrix, m_Size, m_Size, j);
		}
		if (calcTrans)
		{
			// If transformation matrix is needed, multiply all columns
			for (j = 0; j < m_Size; j++)
			{
				pHMatrix->leftMultiply(m_TransMatrix, m_Size, m_Size, j);
			}
		}
		for (j = leftStart; j < leftEnd; j++)
		{
			pHMatrix->leftMultiply(m_Matrix, m_Size, m_Size, j);
		}
	}
	else
	{
		// Passing data to threads
		TDataPtr rightDataPtr = new ThreadingDatta[m_Thread];
		TDataPtr leftDataPtr = new ThreadingDatta[m_Thread];
		if (calcTrans)
		{
			// Allocate threads
			transThread = m_Size * m_Thread / (m_Size + leftEnd - leftStart
				+ rightEnd - rightStart);
			if (m_Size > leftEnd - leftStart + rightEnd - rightStart)
			{
				transThread = std::min(transThread + 1, m_Thread);
			}
			matrixThread = m_Thread - transThread;
		}
		else
		{
			matrixThread = m_Thread;
			transThread = 0;
		}
		minPos = rightStart;
		// If number of rows is too small, use main thread only
		if (matrixThread != 0 && rightEnd - rightStart >= threadLimit * matrixThread)
		{
			nRuns = (rightEnd - rightStart) / matrixThread;
			for (j = 0; j < matrixThread; j++)
			{
				// Send data to thread
				rightDataPtr[j].byHouse = true;
				rightDataPtr[j].gPtr = nullptr;
				rightDataPtr[j].colSize = m_Size;
				rightDataPtr[j].rowSize = m_Size;
				rightDataPtr[j].dMatrix = m_Matrix;
				rightDataPtr[j].hPtr = pHMatrix;
				rightDataPtr[j].minLocation = minPos;
				minPos += nRuns;
				rightDataPtr[j].maxLocation = minPos;
				rightMainPromise[j].set_value(&(rightDataPtr[j]));
			}
		}
		// Perform remainding multipications
		for (j = minPos; j < rightEnd; j++)
		{
			pHMatrix->rightMultiply(m_Matrix, m_Size, m_Size, j);
		}
		// If transformation matrix is needed, calculate
		if (calcTrans)
		{
			minPos = 0;
			if (transThread != 0 && m_Size >= threadLimit * transThread)
			{
				nRuns = m_Size / transThread;
				for (j = matrixThread; j < m_Thread; j++)
				{
					leftDataPtr[j].byHouse = true;
					leftDataPtr[j].gPtr = nullptr;
					leftDataPtr[j].colSize = m_Size;
					leftDataPtr[j].rowSize = m_Size;
					leftDataPtr[j].hPtr = pHMatrix;
					leftDataPtr[j].dMatrix = m_TransMatrix;
					leftDataPtr[j].minLocation = minPos;
					minPos += nRuns;
					leftDataPtr[j].maxLocation = minPos;
					leftMainPromise[j].set_value(&(leftDataPtr[j]));
				}
			}
			for (j = minPos; j < m_Size; j++)
			{
				pHMatrix->leftMultiply(m_TransMatrix, m_Size, m_Size, j);
			}
		}
		// Wait for right thread to complete
		if (matrixThread != 0 && (rightEnd - rightStart) >= threadLimit * matrixThread)
		{
			for (j = 0; j < matrixThread; j++)
			{
				std::future<void> rightFuture = rightThreadPromise[j].get_future();
				rightFuture.get();
				std::promise<void> temp;
				rightThreadPromise[j] = std::move(temp);
			}
		}
		// Left multiplication
		minPos = leftStart;
		if (matrixThread != 0 && (leftEnd - leftStart) >= threadLimit * matrixThread)
		{
			nRuns = (leftEnd - leftStart) / matrixThread;
			for (MInt j = 0; j < matrixThread; j++)
			{
				leftDataPtr[j].byHouse = true;
				leftDataPtr[j].gPtr = nullptr;
				leftDataPtr[j].colSize = m_Size;
				leftDataPtr[j].rowSize = m_Size;
				leftDataPtr[j].hPtr = pHMatrix;
				leftDataPtr[j].dMatrix = m_Matrix;
				leftDataPtr[j].minLocation = minPos;
				minPos += nRuns;
				leftDataPtr[j].maxLocation = minPos;
				leftMainPromise[j].set_value(&(leftDataPtr[j]));
			}
		}
		for (j = minPos; j < leftEnd; j++)
		{
			pHMatrix->leftMultiply(m_Matrix, m_Size, m_Size, j);
		}
		// Wait for transformation matrix multiplication to complete
		if (calcTrans)
		{
			if (transThread != 0 && m_Size >= threadLimit * transThread)
			{
				for (j = matrixThread; j < m_Thread; j++)
				{
					std::future<void> leftFuture = leftThreadPromise[j].get_future();
					leftFuture.get();
					std::promise<void> temp;
					leftThreadPromise[j] = std::move(temp);
				}
			}
		}
		// Wait for left multiplication to complete
		if (matrixThread != 0 && (leftEnd - leftStart) >= threadLimit * matrixThread)
		{
			for (j = 0; j < matrixThread; j++)
			{
				std::future<void> leftFuture = leftThreadPromise[j].get_future();
				leftFuture.get();
				std::promise<void> temp;
				leftThreadPromise[j] = std::move(temp);
			}
		}
		delete[] rightDataPtr;
		delete[] leftDataPtr;
	}
}

// Perform matrix multiplication on transformation matrix
void DenseMatrix::multiplyByGivens(
	std::promise<TDataPtr>* leftMainPromise,
	std::promise<void>* leftThreadPromise,
	GivensPtr pGivens)
{
	if (m_Thread == 0)
	{
		// Perform all multiplication on main thread
		for (MInt i = 0; i < m_Size; i++)
		{
			pGivens->leftMultiply(m_TransMatrix, m_Size, m_Size, i);
		}
	}
	else
	{
		// Perform multiplication on threads
		TDataPtr dataPtr = new ThreadingDatta[m_Thread];
		MInt minPos = 0;
		if (m_Size >= threadLimit * m_Thread)
		{
			MInt nRuns = m_Size / m_Thread;
			for (MInt i = 0; i < m_Thread; i++)
			{
				dataPtr[i].byHouse = false;
				dataPtr[i].hPtr = nullptr;
				dataPtr[i].gPtr = pGivens;
				dataPtr[i].dMatrix = m_TransMatrix;
				dataPtr[i].colSize = m_Size;
				dataPtr[i].rowSize = m_Size;
				dataPtr[i].minLocation = minPos;
				minPos += nRuns;
				dataPtr[i].maxLocation = minPos;
				leftMainPromise[i].set_value(&(dataPtr[i]));
			}
		}
		for (MInt i = minPos; i < m_Size; i++)
		{
			pGivens->leftMultiply(m_TransMatrix, m_Size, m_Size, i);
		}
		if (m_Size >= threadLimit * m_Thread)
		{
			for (MInt i = 0; i < m_Thread; i++)
			{
				std::future<void> leftFuture = leftThreadPromise[i].get_future();
				leftFuture.get();
				std::promise<void> temp;
				leftThreadPromise[i] = std::move(temp);
			}
		}
		delete[] dataPtr;
	}
}

// The method first reduce matrix to upper Hessenberg form
// And then reduce to real schur form
void DenseMatrix::schurDecomposeByFrancisQR()
{
	toUpperHessenberg();
	std::promise<TDataPtr>* leftMainPromise = nullptr;
	std::promise<void>* leftThreadPromise = nullptr;
	std::thread* leftThread = nullptr;
	std::promise<TDataPtr>* rightMainPromise = nullptr;
	std::promise<void>* rightThreadPromise = nullptr;
	std::thread* rightThread = nullptr;
	if (m_Thread > 0)
	{
		// Create threads
		leftMainPromise = new std::promise<TDataPtr>[m_Thread];
		leftThreadPromise = new std::promise<void>[m_Thread];
		leftThread = new std::thread[m_Thread];
		rightMainPromise = new std::promise<TDataPtr>[m_Thread];
		rightThreadPromise = new std::promise<void>[m_Thread];
		rightThread = new std::thread[m_Thread];
		for (MInt i = 0; i < m_Thread; i++)
		{
			leftThread[i] = std::thread(leftMultiplyByTransMatrix, leftMainPromise,
				leftThreadPromise, m_Thread, i);
			rightThread[i] = std::thread(rightMultiplyByTransMatrix, rightMainPromise,
				rightThreadPromise, m_Thread, i);
		}
	}
	LDouble dSum;
	LDouble dProduct;
	LDouble dX;
	LDouble tempX;
	LDouble dY;
	LDouble tempY;
	LDouble dZ;
	LDouble tempZ;
	MInt rightLength;
	HousePtr pHMatrix;
	MInt nStart;
	MInt nEnd = m_Size - 1;
	// Use i and j as counter
	MInt i, j;
	while (nEnd >= 2)
	{
		// If elements are too small, set them to zero
		for (j = 1; j <= nEnd; j++)
		{
			if (abs(m_Matrix[j - 1][j]) <
				std::max((abs(m_Matrix[j][j]) + abs(m_Matrix[j - 1][j - 1])) * NUMERIC_ERROR,
					NUMERIC_ERROR))
			{
				m_Matrix[j - 1][j] = 0.0L;
			}
		}
		// Find ending point of non real Schur part
		while (nEnd >= 2 && (abs(m_Matrix[nEnd - 1][nEnd]) <  NUMERIC_ERROR ||
			abs(m_Matrix[nEnd - 2][nEnd - 1]) <  NUMERIC_ERROR))
		{
			nEnd--;
		}
		// If nEnd < 2, the matrix is already a real Schur form
		if (nEnd < 2)
		{
			break;
		}
		// Find starting point of non real Schur part
		nStart = nEnd - 2;
		while (nStart >= 1 && abs(m_Matrix[nStart - 1][nStart]) >=  NUMERIC_ERROR)
		{
			nStart--;
		}
		// Calculate the initial Householder
		dSum = m_Matrix[nEnd - 1][nEnd - 1] + m_Matrix[nEnd][nEnd];
		dProduct = m_Matrix[nEnd - 1][nEnd - 1] * m_Matrix[nEnd][nEnd]
			- m_Matrix[nEnd - 1][nEnd] * m_Matrix[nEnd][nEnd - 1];
		dX = m_Matrix[nStart][nStart] * m_Matrix[nStart][nStart]
			+ m_Matrix[nStart + 1][nStart] * m_Matrix[nStart][nStart + 1]
			- dSum * m_Matrix[nStart][nStart] + dProduct;
		dY = m_Matrix[nStart][nStart + 1]
			* (m_Matrix[nStart][nStart] + m_Matrix[nStart + 1][nStart + 1] - dSum);
		dZ = m_Matrix[nStart][nStart + 1] * m_Matrix[nStart + 1][nStart + 2];
		tempX = m_Matrix[nStart][nStart];
		tempY = m_Matrix[nStart][nStart + 1];
		tempZ = m_Matrix[nStart][nStart + 2];
		m_Matrix[nStart][nStart] = dX;
		m_Matrix[nStart][nStart + 1] = dY;
		m_Matrix[nStart][nStart + 2] = dZ;
		pHMatrix = new Householder(m_Matrix, m_Size, m_Size, nStart, 3, nStart, true);	
		m_Matrix[nStart][nStart] = tempX;
		m_Matrix[nStart][nStart + 1] = tempY;
		m_Matrix[nStart][nStart + 2] = tempZ;
		rightLength = std::min(nEnd + 1, nStart + 5);
		multiplyByHouseholder(
			leftMainPromise,
			leftThreadPromise,
			rightMainPromise,
			rightThreadPromise,
			pHMatrix,
			nStart,
			m_Size,
			(MInt)0,
			rightLength,
			true);
		delete pHMatrix;
		// Multiply a series of Householders
		for (i = nStart; i < nEnd - 2; i++)
		{

			pHMatrix = new Householder(m_Matrix, m_Size, m_Size, i, 3, i + 1, true);
			rightLength = std::min(nEnd + 1, i + 5);
			multiplyByHouseholder(
				leftMainPromise,
				leftThreadPromise,
				rightMainPromise,
				rightThreadPromise,
				pHMatrix,
				i + 1,
				m_Size,
				(MInt)0,
				rightLength,
				true);
			m_Matrix[i][i + 1] = (-1.0L) * pHMatrix->getAlpha();
			m_Matrix[i][i + 2] = 0.0L;
			m_Matrix[i][i + 3] = 0.0L;
			delete pHMatrix;
		}
		pHMatrix = new Householder(m_Matrix, m_Size, m_Size, nEnd - 2, 2, nEnd - 1, true);
		rightLength = nEnd + 1;
		multiplyByHouseholder(
			leftMainPromise,
			leftThreadPromise,
			rightMainPromise,
			rightThreadPromise,
			pHMatrix,
			nEnd - 1,
			m_Size,
			(MInt)0,
			nEnd + 1,
			true);
		m_Matrix[nEnd - 2][nEnd - 1] = (-1.0L) * pHMatrix->getAlpha();
		m_Matrix[nEnd - 2][nEnd] = 0.0L;
		delete pHMatrix;
	}
	if (m_Thread > 0)
	{
		ThreadingDatta hData;
		hData.dMatrix = nullptr;
		hData.byHouse = true;
		hData.hPtr = nullptr;
		hData.gPtr = nullptr;
		for (j = 0; j < m_Thread; j++)
		{
			leftMainPromise[j].set_value(&(hData));
			leftThread[j].join();
			rightMainPromise[j].set_value(&(hData));
			rightThread[j].join();
		}
		if (leftMainPromise)
		{
			delete[] leftMainPromise;
		}
		if (leftThreadPromise)
		{
			delete[] leftThreadPromise;
		}
		if (leftThread)
		{
			delete[] leftThread;
		}
		if (rightMainPromise)
		{
			delete[] rightMainPromise;
		}
		if (rightThreadPromise)
		{
			delete[] rightThreadPromise;
		}
		if (rightThread)
		{
			delete[] rightThread;
		}
	}
}

// Assuming the matrix is symmetrical such that certain rows need not to
//		be multiplied by Householders when reducing matrix to upper
//		Hessenberg
// The remaining operations are performed by multiplying Givens matrix
void DenseMatrix::schurDecomposeBySymmetricQR()
{
	m_Symmetrical = true;
	toUpperHessenberg();
	std::promise<TDataPtr>* leftMainPromise = nullptr;
	std::promise<void>* leftThreadPromise = nullptr;
	std::thread* leftThread = nullptr;
	if (m_Thread > 0)
	{
		leftMainPromise = new std::promise<TDataPtr>[m_Thread];
		leftThreadPromise = new std::promise<void>[m_Thread];
		leftThread = new std::thread[m_Thread];
		for (MInt i = 0; i < m_Thread; i++)
		{
			leftThread[i] = std::thread(leftMultiplyByTransMatrix, leftMainPromise,
				leftThreadPromise, m_Thread, i);
		}
	}
	LDouble dDelta;
	LDouble dU;
	LDouble dX;
	LDouble tempX;
	GivensPtr pGivens;
	MInt nStart;
	MInt nEnd = m_Size - 1;
	while (nEnd > 0)
	{
		for (MInt i = 1; i <= nEnd; i++)
		{
			if (abs(m_Matrix[i - 1][i]) <
				std::max((abs(m_Matrix[i - 1][i - 1]) + abs(m_Matrix[i][i])) * NUMERIC_ERROR,
					NUMERIC_ERROR))
			{
				m_Matrix[i - 1][i] = 0.0L;
				m_Matrix[i][i - 1] = 0.0L;
			}
		}
		// Find ending point of non real Schur part
		while (nEnd > 0 && abs(m_Matrix[nEnd - 1][nEnd]) < NUMERIC_ERROR)
		{
			nEnd--;
		}
		if (nEnd == 0)
		{
			break;
		}
		// Find starting point of non real Schur part
		nStart = nEnd - 1;
		while (nStart > 0 && abs(m_Matrix[nStart - 1][nStart]) >= NUMERIC_ERROR)
		{
			nStart--;
		}
		dDelta = (m_Matrix[nEnd - 1][nEnd - 1] - m_Matrix[nEnd][nEnd]) / (2.0L);
		if (abs(dDelta) >= NUMERIC_ERROR)
		{
			dU = m_Matrix[nEnd][nEnd] - m_Matrix[nEnd - 1][nEnd] * m_Matrix[nEnd - 1][nEnd]
				/ (dDelta + dDelta / abs(dDelta)
					* sqrtl(dDelta * dDelta + m_Matrix[nEnd - 1][nEnd] * m_Matrix[nEnd - 1][nEnd]));
		}
		else
		{
			dU = m_Matrix[nEnd][nEnd] - m_Matrix[nEnd - 1][nEnd];
		}
		dX = m_Matrix[nStart][nStart] - dU;
		tempX = m_Matrix[nStart][nStart];
		m_Matrix[nStart][nStart] = dX;
		pGivens = new Givens(m_Matrix, m_Size, m_Size, nStart, nStart + 1, nStart, true);
		m_Matrix[nStart][nStart] = tempX;
		// Left multiplication of transformation matrix
		multiplyByGivens(leftMainPromise, leftThreadPromise, pGivens);
		// Left multiply by Givens matrix
		pGivens->leftMultiply(m_Matrix, m_Size, m_Size, nStart);
		pGivens->leftMultiply(m_Matrix, m_Size, m_Size, nStart + 1);
		if (nStart < nEnd - 1)
		{
			pGivens->leftMultiply(m_Matrix, m_Size, m_Size, nStart + 2);
		}
		// Right multiply by transpose of Givens matrix
		pGivens->rightMultiply(m_Matrix, m_Size, m_Size, nStart);
		pGivens->rightMultiply(m_Matrix, m_Size, m_Size, nStart + 1);
		if (nStart < nEnd - 1)
		{
			pGivens->rightMultiply(m_Matrix, m_Size, m_Size, nStart + 2);
		}
		delete pGivens;
		if (nEnd > 2)
		{
			for (MInt i = nStart; i < nEnd - 2; i++)
			{
				pGivens = new Givens(m_Matrix, m_Size, m_Size, i, i + 2, i + 1, true);
				// Left multiplication of transformation matrix
				multiplyByGivens(leftMainPromise, leftThreadPromise, pGivens);
				// Left multiply by Givens matrix
				pGivens->leftMultiply(m_Matrix, m_Size, m_Size, i);
				pGivens->leftMultiply(m_Matrix, m_Size, m_Size, i + 1);
				pGivens->leftMultiply(m_Matrix, m_Size, m_Size, i + 2);
				pGivens->leftMultiply(m_Matrix, m_Size, m_Size, i + 3);
				// Right multiply by transpose of Givens matirx
				pGivens->rightMultiply(m_Matrix, m_Size, m_Size, i);
				pGivens->rightMultiply(m_Matrix, m_Size, m_Size, i + 1);
				pGivens->rightMultiply(m_Matrix, m_Size, m_Size, i + 2);
				pGivens->rightMultiply(m_Matrix, m_Size, m_Size, i + 3);
				delete pGivens;
			}
		}
		if (nEnd > 1)
		{
			pGivens = new Givens(m_Matrix, m_Size, m_Size, nEnd - 2, nEnd, nEnd - 1, true);
			multiplyByGivens(leftMainPromise, leftThreadPromise, pGivens);
			pGivens->leftMultiply(m_Matrix, m_Size, m_Size, nEnd - 2);
			pGivens->leftMultiply(m_Matrix, m_Size, m_Size, nEnd - 1);
			pGivens->leftMultiply(m_Matrix, m_Size, m_Size, nEnd);
			pGivens->rightMultiply(m_Matrix, m_Size, m_Size, nEnd - 2);
			pGivens->rightMultiply(m_Matrix, m_Size, m_Size, nEnd - 1);
			pGivens->rightMultiply(m_Matrix, m_Size, m_Size, nEnd);
			delete pGivens;
		}
	}
	if (m_Thread > 0)
	{
		ThreadingDatta hData;
		hData.dMatrix = nullptr;
		hData.byHouse = true;
		hData.hPtr = nullptr;
		hData.gPtr = nullptr;
		for (MInt i = 0; i < m_Thread; i++)
		{
			leftMainPromise[i].set_value(&(hData));
			leftThread[i].join();
		}
		if (leftMainPromise)
		{
			delete[] leftMainPromise;
		}
		if (leftThreadPromise)
		{
			delete[] leftThreadPromise;
		}
		if (leftThread)
		{
			delete[] leftThread;
		}
	}
}

// Transform matrix to upper triangle form using multiple threads
void DenseMatrix::toUpperTriangleByHouseholder(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Pointer to constant vector cannot be nullptr.");
	}
	if (m_Size != nSize)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	std::promise<TDataPtr>* mainPromise = nullptr;
	std::promise<void>* threadPromise = nullptr;
	std::thread* pThread = nullptr;
	LDPtr* vectorMatrix = new LDPtr[1];
	vectorMatrix[0] = pVector;
	HousePtr pHouse;
	if (m_Thread > 0)
	{
		mainPromise = new std::promise<TDataPtr>[m_Thread];
		threadPromise = new std::promise<void>[m_Thread];
		pThread = new std::thread[m_Thread];
		for (MInt i = 0; i < m_Thread; i++)
		{
			pThread[i] = std::thread(leftMultiplyByTransMatrix, mainPromise,
				threadPromise, m_Thread, i);
		}
	}
	if (m_Size <= 1)
	{
		return;
	}
	MInt j;
	for (MInt i = 0; i < m_Size - 1; i++)
	{
		pHouse = new Householder(m_Matrix, m_Size, m_Size, i, m_Size - i, true);
		multiplyByHouseholder(
			mainPromise,
			threadPromise,
			nullptr,
			nullptr,
			pHouse,
			i + 1,
			m_Size,
			(MInt)0,
			(MInt)0,
			false);
		// Set element to alpha
		m_Matrix[i][i] = (-1.0L) * (pHouse->getAlpha());
		// Set the rest to zero
		for (j = i + 1; j < m_Size; j++)
		{
			m_Matrix[i][j] = 0.0L;
		}
		pHouse->leftMultiply(vectorMatrix, m_Size, 1, 0);
		delete pHouse;
	}
	ThreadingDatta hData;
	hData.dMatrix = nullptr;
	hData.byHouse = true;
	hData.gPtr = nullptr;
	hData.hPtr = nullptr;
	for (j = 0; j < m_Thread; j++)
	{
		mainPromise[j].set_value(&(hData));
		pThread[j].join();
	}
	if (m_Thread > 0)
	{
		if (mainPromise)
		{
			delete[] mainPromise;
		}
		if (threadPromise)
		{
			delete[] threadPromise;
		}
		if (pThread)
		{
			delete[] pThread;
		}
	}
	delete[] vectorMatrix;
}

// Solve a system of linear equations
bool DenseMatrix::solvedByHouseholder(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Pointer to constant vector is nullptr.");
	}
	if (m_Size != nSize)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	toUpperTriangleByHouseholder(pVector, nSize);
	return solveUpperTriangle(pVector, nSize);
}

// Reduce matrix to the product of a not-singular lower trianble matrix and this
//		matrix's transpose, assuming the original matrix is positively-defined
// The transpose is stored in the upper triangle part
bool DenseMatrix::choleskyDecompose()
{
	LDouble dSum;
	LDouble dSqrSum;
	for (MInt i = 0; i < m_Size; i++)
	{
		dSqrSum = 0.0L;
		for (MInt j = 0; j < i; j++)
		{
			dSum = 0.0L;
			for (MInt k = 0; k < j; k++)
			{
				dSum += (m_Matrix[k][i] * m_Matrix[k][j]);
			}
			m_Matrix[j][i] = (m_Matrix[j][i] - dSum) / m_Matrix[j][j];
			m_Matrix[i][j] = m_Matrix[j][i];
			dSqrSum += (m_Matrix[j][i] * m_Matrix[j][i]);
		}
		if (m_Matrix[i][i] - dSqrSum < NUMERIC_ERROR)
		{
			return false;
		}
		m_Matrix[i][i] = sqrtl(m_Matrix[i][i] - dSqrSum);
	}
	return true;
}

// Reduce to Cholesky form and solve two systems of linear equations
bool DenseMatrix::solvedByCholesky(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Constant vector cannot be nullptr.");
	}
	if (m_Size != nSize)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	if (choleskyDecompose())
	{
		if (!solveLowerTriangle(pVector, nSize))
		{
			return false;
		}
		if (!solveUpperTriangle(pVector, nSize))
		{
			return false;
		}
		return true;
	}
	else
	{
		return false;
	}
}

// This method is used to check whether toUpperHessenberg() generates the correct
//		result assuming the matrix is symmetrical
bool DenseMatrix::solveTriDiagonal(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Constant vector cannot be nullptr.");
	}
	if (m_Size != nSize)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	LDPtr* vectorMatrix = new LDPtr[1];
	vectorMatrix[0] = pVector;
	for (MInt i = 0; i < m_Size - 2; i++)
	{
		Householder hMatrix(m_Matrix, m_Size, m_Size, i, 2, i, true);
		hMatrix.leftMultiply(m_Matrix, m_Size, m_Size, i + 1);
		hMatrix.leftMultiply(m_Matrix, m_Size, m_Size, i + 2);
		hMatrix.leftMultiply(vectorMatrix, m_Size, 1, 0);
		m_Matrix[i][i] = (-1.0L) * hMatrix.getAlpha();
		m_Matrix[i][i + 1] = 0.0L;
	}

	if (m_Size > 1)
	{
		Householder hMatrix(m_Matrix, m_Size, m_Size, m_Size - 2, 2, m_Size - 2, true);
		hMatrix.leftMultiply(m_Matrix, m_Size, m_Size, m_Size - 1);
		hMatrix.leftMultiply(vectorMatrix, m_Size, 1, 0);
		m_Matrix[m_Size - 2][m_Size - 2] = (-1.0L) * hMatrix.getAlpha();
		m_Matrix[m_Size - 2][m_Size - 1] = 0.0L;

	}
	delete[] vectorMatrix;
	return solveUpperTriDiagonal(pVector, nSize);
}

// Difference class methods will call this methods
// Add here to avoid copying memory
bool externalSolveUpperTriangle(LDPtr* pMatrix, LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Pointer to constant vector cannot be nullptr.");
	}
	LDouble dSum;
	for (int i = (int)nSize - 1; i >= 0; i--)
	{
		dSum = 0.0L;
		for (int j = (int)nSize - 1; j > i; j--)
		{
			dSum += (pMatrix[j][i] * pVector[j]);
		}

		if (abs(pMatrix[i][i]) < NUMERIC_ERROR)
		{
			return false;
		}
		pVector[i] = (pVector[i] - dSum) / pMatrix[i][i];
	}
	return true;
}

bool DenseMatrix::solveUpperTriangle(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Pointer to constant vector cannot be nullptr.");
	}
	if (m_Size != nSize)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	return externalSolveUpperTriangle(m_Matrix, pVector, m_Size);
}

// This method is used to check the result of toUpperHessenberg() assuming matrix is
//		symmetrical
bool DenseMatrix::solveUpperTriDiagonal(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Constant vector cannot be nullptr.");
	}
	if (m_Size != nSize)
	{
		throw std::invalid_argument("Sizes are not match.");
	}
	LDouble dSum;
	if (m_Size >= 1 && abs(m_Matrix[m_Size - 1][m_Size - 1]) < NUMERIC_ERROR)
	{
		return false;
	}
	pVector[m_Size - 1] = pVector[m_Size - 1] / m_Matrix[m_Size - 1][m_Size - 1];

	if (m_Size >= 2 && abs(m_Matrix[m_Size - 2][m_Size - 2]) < NUMERIC_ERROR)
	{
		return false;
	}
	dSum = m_Matrix[m_Size - 1][m_Size - 2] * pVector[m_Size - 1];
	pVector[m_Size - 2] = (pVector[m_Size - 2] - dSum) / m_Matrix[m_Size - 2][m_Size - 2];
	
	for (int i = (int)m_Size - 3; i >= 0; i--)
	{
		dSum = m_Matrix[i + 1][i] * pVector[i + 1];
		dSum += (m_Matrix[i + 2][i] * pVector[i + 2]);
		if (abs(m_Matrix[i][i]) < NUMERIC_ERROR)
		{
			return false;
		}
		pVector[i] = (pVector[i] - dSum) / m_Matrix[i][i];
	}
	return true;
}

// Assuming that matrix is of lower triangle form and solve system of linear equations
bool DenseMatrix::solveLowerTriangle(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Constant vector cannot be nullptr.");
	}
	if (nSize != m_Size)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	LDouble dSum;
	for (MInt i = 0; i < m_Size; i++)
	{
		dSum = 0.0L;
		for (MInt j = 0; j < i; j++)
		{
			dSum += (m_Matrix[j][i] * pVector[j]);
		}
		if (abs(m_Matrix[i][i]) < NUMERIC_ERROR)
		{
			return false;
		}
		pVector[i] = (pVector[i] - dSum) / m_Matrix[i][i];
	}
	return true;
}


GeneralMatrix::GeneralMatrix(LDPtr* pMatrix, MInt colSize, MInt rowSize)
{
	if (colSize == 0 || rowSize == 0)
	{
		throw std::invalid_argument("Size cannot be zero.");
	}
	if (colSize < rowSize)
	{
		throw std::invalid_argument("Column size has to be greater than or equal to row size.");
	}
	bool memoryError = false;
	m_ColSize = colSize;
	m_RowSize = rowSize;
	m_Thread = NUMBER_OF_THREADING;
	m_LeftTransMatrix = nullptr;
	m_Matrix = nullptr;
	m_RightTransMatrix = nullptr;
	m_LeftTransMatrix = new LDPtr[colSize];
	// If memory allocation fails, clear all memory created and throw std::bad_alloc()
	if (!m_LeftTransMatrix)
	{
		memoryError = true;
		// Use goto to avoid keeping track of memory that has been created
		goto CleanUpMemory;
	}
	m_Matrix = new LDPtr[rowSize];
	if (!m_Matrix)
	{
		memoryError = true;
		goto CleanUpMemory;
	}
	m_RightTransMatrix = new LDPtr[rowSize];
	if (!m_RightTransMatrix)
	{
		memoryError = true;
		goto CleanUpMemory;
	}
	// Left transformation matrix is colSize x colSize matrix
	// Right transformation matrixx is rowSize x rowSize matrix
	for (MInt i = 0; i < rowSize; i++)
	{
		m_LeftTransMatrix[i] = nullptr;
		m_Matrix[i] = nullptr;
		m_RightTransMatrix[i] = nullptr;
	}
	for (MInt i = rowSize; i < colSize; i++)
	{
		m_LeftTransMatrix[i] = nullptr;
	}
	for (MInt i = 0; i < rowSize; i++)
	{
		m_LeftTransMatrix[i] = new LDouble[colSize];
		if (!m_LeftTransMatrix[i])
		{
			memoryError = true;
			goto CleanUpMemory;
		}
		m_Matrix[i] = new LDouble[colSize];
		if (!m_Matrix[i])
		{
			memoryError = true;
			goto CleanUpMemory;
		}
		m_RightTransMatrix[i] = new LDouble[rowSize];
		if (!m_RightTransMatrix[i])
		{
			memoryError = true;
			goto CleanUpMemory;
		}
		if (!pMatrix || !pMatrix[i])
		{
			for (MInt j = 0; j < i; j++)
			{
				m_LeftTransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = 0.0L;
				m_RightTransMatrix[i][j] = 0.0L;
			}
			m_LeftTransMatrix[i][i] = 1.0L;
			m_Matrix[i][i] = 0.0L;
			m_RightTransMatrix[i][i] = 1.0L;
			for (MInt j = i + 1; j < rowSize; j++)
			{
				m_LeftTransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = 0.0L;
				m_RightTransMatrix[i][j] = 0.0L;
			}
			for (MInt j = rowSize; j < colSize; j++)
			{
				m_LeftTransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = 0.0L;
			}
		}
		else
		{
			for (MInt j = 0; j < i; j++)
			{
				m_LeftTransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = pMatrix[i][j];
				m_RightTransMatrix[i][j] = 0.0L;
			}
			m_LeftTransMatrix[i][i] = 1.0L;
			m_Matrix[i][i] = pMatrix[i][i];
			m_RightTransMatrix[i][i] = 1.0L;
			for (MInt j = i + 1; j < rowSize; j++)
			{
				m_LeftTransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = pMatrix[i][j];
				m_RightTransMatrix[i][j] = 0.0L;
			}
			for (MInt j = rowSize; j < colSize; j++)
			{
				m_LeftTransMatrix[i][j] = 0.0L;
				m_Matrix[i][j] = pMatrix[i][j];
			}
		}
	}
	for (MInt i = rowSize; i < colSize; i++)
	{
		m_LeftTransMatrix[i] = new LDouble[colSize];
		if (!m_LeftTransMatrix[i])
		{
			memoryError = true;
			goto CleanUpMemory;
		}
		for (MInt j = 0; j < i; j++)
		{
			m_LeftTransMatrix[i][j] = 0.0L;
		}
		m_LeftTransMatrix[i][i] = 1.0L;
		for (MInt j = i + 1; j < colSize; j++)
		{
			m_LeftTransMatrix[i][j] = 0.0L;
		}
	}
CleanUpMemory:
	if (memoryError)
	{
		clearMemory();
		throw std::bad_alloc();
	}
}

void GeneralMatrix::clearMemory()
{
	if (m_LeftTransMatrix)
	{
		for (MInt i = 0; i < m_ColSize; i++)
		{
			if (m_LeftTransMatrix[i])
			{
				delete[] m_LeftTransMatrix[i];
				m_LeftTransMatrix[i] = nullptr;
			}
		}
		delete[] m_LeftTransMatrix;
		m_LeftTransMatrix = nullptr;
	}
	if (m_RightTransMatrix)
	{
		for (MInt i = 0; i < m_RowSize; i++)
		{
			if (m_RightTransMatrix[i])
			{
				delete[] m_RightTransMatrix[i];
				m_RightTransMatrix[i] = nullptr;
			}
		}
		delete[] m_RightTransMatrix;
		m_RightTransMatrix = nullptr;
	}
	if (m_Matrix)
	{
		for (MInt i = 0; i < m_RowSize; i++)
		{
			if (m_Matrix[i])
			{
				delete[] m_Matrix[i];
				m_Matrix[i] = nullptr;
			}
		}
		delete[] m_Matrix;
		m_Matrix = nullptr;
	}
}

GeneralMatrix::~GeneralMatrix()
{
	clearMemory();
}

// Sum up square of column elements for several columns, whose positions
//		range from nStart to nEnd, not including nEnd
// To reduce numeric error, a column is divided into several intervals,
//		sum up square of these intervals, and then sum up the resulting sums
// Using this function for multithreading
// Parameters:
//		dMatrix, pointer to matrix
//		dVector, vector to hold square sum of each columns
//		colSize, the size of column
void sumColumns(LDPtr* dMatrix, LDPtr dVector, MInt colSize,
	MInt nStart, MInt nEnd)
{
	MInt intervals = colSize / VECTOR_INTERVAL_SIZE;
	intervals++;
	MInt intervalSize = colSize / intervals;
	intervalSize++;
	LDouble dSum;
	MInt counter;
	for (MInt i = nStart; i < nEnd; i++)
	{
		dVector[i] = 0.0L;
		for (MInt j = 0; j < intervals - 1; j++)
		{
			dSum = 0.0L;
			for (counter = 0; counter < intervalSize; counter++)
			{
				dSum += (dMatrix[i][j * intervalSize + counter]
					* dMatrix[i][j * intervalSize + counter]);
			}
			dVector[i] += dSum;
		}
		dSum = 0.0L;
		for (counter = (intervals - 1) * intervalSize; counter < colSize; counter++)
		{
			dSum += (dMatrix[i][counter] * dMatrix[i][counter]);
		}
		dVector[i] += dSum;
	}
}

// This method first reduce matrix to a form like
// * * * * * * * *
// 0 * * * * * * *
// 0 0 * * * * * *
// 0 0 0 * * * * *
// 0 0 0 0 * * * *
// 0 0 0 0 0 0 0 0
// 0 0 0 0 0 0 0 0
// 0 0 0 0 0 0 0 0
// 0 0 0 0 0 0 0 0
// During the reduction process, the columns are exchanged so that the column with
//		larger square sum will be considered first
// The exchange matrix is also recorded
// The vector passed as parameters will also be multiplied by the transformation matrix
// Then reduce it to upper triangle form like
// * * * * * 0 0 0
// 0 * * * * 0 0 0
// 0 0 * * * 0 0 0
// 0 0 0 * * 0 0 0
// 0 0 0 0 * 0 0 0
// 0 0 0 0 0 0 0 0
// 0 0 0 0 0 0 0 0
// 0 0 0 0 0 0 0 0
// 0 0 0 0 0 0 0 0
// Then, solve system of linear equations with resulting dimension and corresponding column elements
// Interchange elements so that elements have original order
bool GeneralMatrix::solveLeastSquare(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Pointer to constant vector cannot be nullptr.");
	}
	if (m_ColSize != nSize)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	// m_ColSize x 1 matrix to hold the vector
	LDPtr* vectorMatrix = nullptr;
	// Exchange matrix
	MInt* exMatrix = nullptr;
	LDPtr intervalSum = nullptr;
	LDPtr rowSum = nullptr;
	// A list of Householder to record
	HousePtr pRightHouse = nullptr;
	// Additional exchange matrix used during the process
	MInt* exchangePre = nullptr;
	MInt* exchangeAft = nullptr;
	// Use to interchange elements of vector
	LDPtr vectorCopy = nullptr;
	// For multithreading
	std::promise<TDataPtr>* mainPromise = nullptr;
	std::promise<void>* threadPromise = nullptr;
	std::thread* pThread = nullptr;
	vectorMatrix = new LDPtr[1];
	vectorMatrix[0] = new LDouble[m_ColSize];
	for (MInt i = 0; i < m_ColSize; i++)
	{
		vectorMatrix[0][i] = pVector[i];
	}
	if (m_Thread > 0)
	{
		mainPromise = new std::promise<TDataPtr>[m_Thread];
		threadPromise = new std::promise<void>[m_Thread];
		pThread = new std::thread[m_Thread];
		for (MInt i = 0; i < m_Thread; i++)
		{
			pThread[i] = std::thread(leftMultiplyByTransMatrix, mainPromise,
				threadPromise, m_Thread, i);
		}
	}
	exMatrix = new MInt[m_RowSize];
	rowSum = new LDouble[m_RowSize];
	pRightHouse = new Householder[m_RowSize];

	// Sum up squares
	if (m_Thread == 0 || m_RowSize < m_Thread * threadLimit)
	{
		MInt position = m_ColSize / VECTOR_INTERVAL_SIZE;
		position++;
		MInt tLimit = m_ColSize / position;
		tLimit++;
		LDouble dSum;
		MInt counter;
		for (MInt i = 0; i < m_RowSize; i++)
		{
			rowSum[i] = 0.0L;
			for (MInt j = 0; j < position - 1; j++)
			{
				dSum = 0.0L;
				for (counter = 0; counter < tLimit; counter++)
				{
					dSum += (m_Matrix[i][j * tLimit + counter]
						* m_Matrix[i][j * tLimit + counter]);
				}
				rowSum[i] += dSum;
			}
			dSum = 0.0L;
			for (counter = (position - 1) * tLimit; counter < m_ColSize; counter++)
			{
				dSum += (m_Matrix[i][counter] * m_Matrix[i][counter]);
			}
			rowSum[i] += dSum;
		}
	}
	else
	{
		std::thread* summingThread = new std::thread[m_Thread];
		MInt taskSize = m_RowSize / m_Thread;
		taskSize++;
		for (MInt i = 0; i < m_Thread - 1; i++)
		{
			summingThread[i] = std::thread(sumColumns, m_Matrix, rowSum, m_ColSize,
				i * taskSize, (i + 1) * taskSize);
		}
		summingThread[m_Thread - 1] = std::thread(sumColumns, m_Matrix, rowSum, m_ColSize,
			(m_Thread - 1) * taskSize, m_RowSize);
		for (MInt i = 0; i < m_Thread; i++)
		{
			summingThread[i].join();
		}
		delete[] summingThread;
	}
	// Error allowed
	LDouble maxDouble = 0.0L;
	for (MInt i = 0; i < m_RowSize; i++)
	{
		exMatrix[i] = i;
		if (maxDouble < rowSum[i])
		{
			maxDouble = rowSum[i];
		}
	}
	LDouble dError = std::max(maxDouble * NUMERIC_ERROR, NUMERIC_ERROR);
	MInt houseCounter = 0;
	MInt position;
	for (MInt i = 0; i < m_RowSize; i++)
	{
		position = i;
		maxDouble = rowSum[i];
		for (MInt j = i + 1; j < m_RowSize; j++)
		{
			if (maxDouble < rowSum[j])
			{
				maxDouble = rowSum[j];
				position = j;
			}
		}
		if (abs(maxDouble) < dError)
		{
			break;
		}
		if (position != i)
		{
			MInt tempInt = exMatrix[i];
			exMatrix[i] = exMatrix[position];
			exMatrix[position] = tempInt;
			LDouble tempDouble = rowSum[i];
			rowSum[i] = rowSum[position];
			rowSum[position] = tempDouble;
			LDPtr tempPtr = m_Matrix[i];
			m_Matrix[i] = m_Matrix[position];
			m_Matrix[position] = tempPtr;
		}
		// Householder transformation
		Householder hMatrix(m_Matrix, m_ColSize, m_RowSize, i, m_ColSize - i, true);
		houseCounter++;
		multiplyByHouseholder(
			mainPromise,
			threadPromise,
			&(hMatrix),
			i + 1,
			m_RowSize,
			true,
			false);
		m_Matrix[i][i] = (-1.0L) * hMatrix.getAlpha();
		// Householder transfromation on vector
		hMatrix.leftMultiply(vectorMatrix, m_ColSize, 1, 0);
		for (MInt j = i + 1; j < m_ColSize; j++)
		{
			m_Matrix[i][j] = 0.0L;
		}
		for (MInt j = i + 1; j < m_RowSize; j++)
		{
			rowSum[j] -= (m_Matrix[j][i] * m_Matrix[j][i]);
		}
	}
	// Terminate all threads
	ThreadingDatta hData;
	hData.dMatrix = nullptr;
	hData.byHouse = true;
	hData.gPtr = nullptr;
	hData.hPtr = nullptr;
	for (MInt j = 0; j < m_Thread; j++)
	{
		mainPromise[j].set_value(&(hData));
		pThread[j].join();
	}
	exchangePre = new MInt[m_RowSize];
	exchangeAft = new MInt[m_RowSize];
	for (MInt i = 0; i < m_RowSize; i++)
	{
		exchangePre[i] = i;
		exchangeAft[i] = i;
	}
	if (m_RowSize - houseCounter > 0)
	{
		// Interchange columns
		LDPtr* tempMatrix = new LDPtr[m_RowSize];
		for (MInt i = 0; i < m_RowSize; i++)
		{
			tempMatrix[i] = m_Matrix[m_RowSize - 1 - i];
			exchangePre[i] = m_RowSize - 1 - i;
		}
		delete[] m_Matrix;
		m_Matrix = tempMatrix;
		// Use Hosueholder to change unwanted elements to be zero
		for (MInt i = 0; i < houseCounter; i++)
		{
			Householder hMatrix(m_Matrix, houseCounter, m_RowSize, houseCounter - 1 - i,
				m_RowSize - houseCounter + 1, i, false);
			pRightHouse[i] = hMatrix;
			m_Matrix[i][houseCounter - 1 - i] = (-1.0L) * hMatrix.getAlpha();
			for (MInt j = i + 1; j < i + m_RowSize - houseCounter + 1; j++)
			{
				m_Matrix[j][houseCounter - 1 - i] = 0.0L;
			}
			for (MInt j = 0; j < houseCounter - 1 - i; j++)
			{
				hMatrix.rightMultiply(m_Matrix, houseCounter, m_RowSize, j);
			}
		}
		tempMatrix = new LDPtr[m_RowSize];

		for (MInt i = 0; i < houseCounter; i++)
		{
			tempMatrix[i] = m_Matrix[houseCounter - 1 - i];
			exchangeAft[i] = houseCounter - 1 - i;
		}
		for (MInt i = houseCounter; i < m_RowSize; i++)
		{
			tempMatrix[i] = m_Matrix[i];
			exchangeAft[i] = i;
		}
		delete[] m_Matrix;
		m_Matrix = tempMatrix;
	}
	// Solve reduced system of linear equations
	externalSolveUpperTriangle(m_Matrix, vectorMatrix[0], houseCounter);
	for (MInt i = houseCounter; i < m_ColSize; i++)
	{
		vectorMatrix[0][i] = 0.0L;
	}
	// Exchange elements
	vectorCopy = new LDouble[m_RowSize];
	for (MInt i = 0; i < m_RowSize; i++)
	{
		vectorCopy[exchangeAft[i]] = vectorMatrix[0][i];
	}
	delete[] vectorMatrix[0];
	vectorMatrix[0] = vectorCopy;
	// Apply Householder on solution
	if (m_RowSize - houseCounter > 0)
	{
		for (MInt i = houseCounter; i > 1; i--)
		{
			pRightHouse[i - 1].leftMultiply(vectorMatrix, m_RowSize, 1, 0);
		}
	}
	vectorCopy = new LDouble[m_RowSize];
	for (MInt i = 0; i < m_RowSize; i++)
	{
		vectorCopy[exchangePre[i]] = vectorMatrix[0][i];
	}
	delete vectorMatrix[0];
	vectorMatrix[0] = vectorCopy;
	vectorCopy = new LDouble[m_RowSize];
	for (MInt i = 0; i < m_RowSize; i++)
	{
		vectorCopy[exMatrix[i]] = vectorMatrix[0][i];
	}
	for (MInt i = 0; i < m_RowSize; i++)
	{
		pVector[i] = vectorCopy[i];
	}
	for (MInt i = m_RowSize; i < m_ColSize; i++)
	{
		pVector[i] = 0.0L;
	}
	if (vectorMatrix)
	{
		if (vectorMatrix[0])
		{
			delete[] vectorMatrix[0];
		}
		delete[] vectorMatrix;
	}
	if (vectorCopy)
	{
		delete[] vectorCopy;
	}
	if (m_Thread > 0)
	{
		if (mainPromise)
		{
			delete[] mainPromise;
		}
		if (threadPromise)
		{
			delete[] threadPromise;
		}
		if (pThread)
		{
			delete[] pThread;
		}
	}
	if (exMatrix)
	{
		delete[] exMatrix;
	}
	if (intervalSum)
	{
		delete[] intervalSum;
	}
	if (rowSum)
	{
		delete[] rowSum;
	}
	if (pRightHouse)
	{
		delete[] pRightHouse;
	}
	if (exchangePre)
	{
		delete[] exchangePre;
	}
	if (exchangeAft)
	{
		delete[] exchangeAft;
	}
	return true;
}

// Assuming the matrix has a rank of m_RowSize
// First reduce it to upper triangle, for example
// * * * * * * *
// 0 * * * * * *
// 0 0 * * * * *
// 0 0 0 * * * *
// 0 0 0 0 * * *
// 0 0 0 0 0 * *
// 0 0 0 0 0 0 *
// 0 0 0 0 0 0 0
// During the process, the vector passed as argument is also multiplied by
//		Householder
// Then solve system of linear equations
bool GeneralMatrix::solveLeastSquareFullRank(LDPtr pVector, MInt nSize)
{
	if (!pVector)
	{
		throw std::invalid_argument("Pointer to constant vector cannot be nullptr.");
	}
	if (m_ColSize != nSize)
	{
		throw std::invalid_argument("Sizes don't match.");
	}
	LDPtr* vectorMatrix = new LDPtr[1];
	vectorMatrix[0] = pVector;
	for (MInt i = 0; i < m_RowSize; i++)
	{
		Householder hMatrix(m_Matrix, m_ColSize, m_RowSize, i, m_ColSize - i, true);
		m_Matrix[i][i] = (-1.0L) * hMatrix.getAlpha();
		for (MInt j = i + 1; j < m_ColSize; j++)
		{
			m_Matrix[i][j] = 0.0L;
		}
		for (MInt j = i + 1; j < m_RowSize; j++)
		{
			hMatrix.leftMultiply(m_Matrix, m_ColSize, m_RowSize, j);
		}
		hMatrix.leftMultiply(vectorMatrix, m_ColSize, 1, 0);
	}
	for (MInt i = m_RowSize; i < m_ColSize; i++)
	{
		vectorMatrix[0][i] = 0.0L;
	}
	bool result = externalSolveUpperTriangle(m_Matrix, vectorMatrix[0], m_RowSize);
	delete[] vectorMatrix;
	return result;
}

// Multiply all required columns or rows by Householder using multiply threads
void GeneralMatrix::multiplyByHouseholder(
	std::promise<TDataPtr>* mainPromise,
	std::promise<void>* threadPromise,
	HousePtr pHouse,
	MInt nStart,
	MInt nEnd,
	bool byLeft,
	bool calcTrans)
{
	MInt j;
	MInt minPos;
	MInt nRuns;
	MInt transThread;
	MInt matrixThread;
	if (nStart > nEnd)
	{
		throw std::invalid_argument(
			"Starting point cannot be greater than ending point.");
	}
	if (byLeft)
	{
		if (m_Thread == 0)
		{
			for (j = nStart; j < nEnd; j++)
			{
				pHouse->leftMultiply(m_Matrix, m_ColSize, m_RowSize, j);
			}
			if (calcTrans)
			{
				for (j = 0; j < m_ColSize; j++)
				{
					pHouse->leftMultiply(m_LeftTransMatrix, m_ColSize, m_ColSize, j);
				}
			}
		}
		else
		{
			TDataPtr dataPtr = new ThreadingDatta[m_Thread];
			if (calcTrans)
			{
				transThread = m_ColSize * m_Thread / (m_ColSize + nEnd - nStart);
				transThread = std::min(transThread + 1, m_Thread);
				matrixThread = m_Thread - transThread;
			}
			else
			{
				matrixThread = m_Thread;
				transThread = 0;
			}
			minPos = nStart;
			if (matrixThread != 0 && (nEnd - nStart) >= threadLimit * matrixThread)
			{
				nRuns = (nEnd - nStart) / matrixThread;
				for (j = 0; j < matrixThread; j++)
				{
					dataPtr[j].byHouse = true;
					dataPtr[j].gPtr = nullptr;
					dataPtr[j].hPtr = pHouse;
					dataPtr[j].colSize = m_ColSize;
					dataPtr[j].rowSize = m_RowSize;
					dataPtr[j].dMatrix = m_Matrix;
					dataPtr[j].minLocation = minPos;
					minPos += nRuns;
					dataPtr[j].maxLocation = minPos;
					mainPromise[j].set_value(&(dataPtr[j]));
				}
			}
			for (j = minPos; j < nEnd; j++)
			{
				pHouse->leftMultiply(m_Matrix, m_ColSize, m_RowSize, j);
			}
			if (calcTrans)
			{
				minPos = 0;
				if (transThread != 0 && m_ColSize >= threadLimit * transThread)
				{
					nRuns = m_ColSize / transThread;
					for (j = matrixThread; j < m_Thread; j++)
					{
						dataPtr[j].byHouse = true;
						dataPtr[j].gPtr = nullptr;
						dataPtr[j].hPtr = pHouse;
						dataPtr[j].colSize = m_ColSize;
						dataPtr[j].rowSize = m_ColSize;
						dataPtr[j].dMatrix = m_LeftTransMatrix;
						dataPtr[j].minLocation = minPos;
						minPos += nRuns;
						dataPtr[j].maxLocation = minPos;
						mainPromise[j].set_value(&(dataPtr[j]));
					}
				}
				for (j = minPos; j < m_ColSize; j++)
				{
					pHouse->leftMultiply(m_LeftTransMatrix, m_ColSize, m_ColSize, j);
				}
				if (transThread != 0 && m_ColSize >= threadLimit * transThread)
				{
					for (j = matrixThread; j < m_Thread; j++)
					{
						std::future<void> threadFuture = threadPromise[j].get_future();
						threadFuture.get();
						std::promise<void> temp;
						threadPromise[j] = std::move(temp);
					}
				}
			}
			if (matrixThread != 0 && (nEnd - nStart) >= threadLimit * matrixThread)
			{
				for (j = 0; j < matrixThread; j++)
				{
					std::future<void> threadFuture = threadPromise[j].get_future();
					threadFuture.get();
					std::promise<void> temp;
					threadPromise[j] = std::move(temp);
				}
			}
			delete[] dataPtr;
		}
	}
	else
	{
		if (m_Thread == 0)
		{
			for (j = nStart; j < nEnd; j++)
			{
				pHouse->rightMultiply(m_Matrix, m_ColSize, m_RowSize, j);
			}
			if (calcTrans)
			{
				for (j = 0; j < m_RowSize; j++)
				{
					pHouse->rightMultiply(m_RightTransMatrix, m_RowSize, m_RowSize, j);
				}
			}
		}
		else
		{
			TDataPtr dataPtr = new ThreadingDatta[m_Thread];
			if (calcTrans)
			{
				transThread = m_RowSize * m_Thread / (m_RowSize + nEnd - nStart);
				transThread = std::min(transThread + 1, m_Thread);
				matrixThread = m_Thread - transThread;
			}
			else
			{
				matrixThread = m_Thread;
				transThread = 0;
			}
			minPos = nStart;
			if (matrixThread != 0 && (nEnd - nStart) >= threadLimit * matrixThread)
			{
				nRuns = (nEnd - nStart) / matrixThread;
				for (j = 0; j < matrixThread; j++)
				{
					dataPtr[j].byHouse = true;
					dataPtr[j].hPtr = pHouse;
					dataPtr[j].gPtr = nullptr;
					dataPtr[j].dMatrix = m_Matrix;
					dataPtr[j].colSize = m_ColSize;
					dataPtr[j].rowSize = m_RowSize;
					dataPtr[j].minLocation = minPos;
					minPos += nRuns;
					dataPtr[j].maxLocation = minPos;
					mainPromise[j].set_value(&(dataPtr[j]));
				}
			}
			for (j = minPos; j < nEnd; j++)
			{
				pHouse->rightMultiply(m_Matrix, m_ColSize, m_RowSize, j);
			}
			if (calcTrans)
			{
				minPos = 0;
				if (transThread != 0 && m_RowSize >= threadLimit * transThread)
				{
					nRuns = m_RowSize / transThread;
					for (j = matrixThread; j < m_Thread; j++)
					{
						dataPtr[j].byHouse = true;
						dataPtr[j].hPtr = pHouse;
						dataPtr[j].gPtr = nullptr;
						dataPtr[j].dMatrix = m_RightTransMatrix;
						dataPtr[j].colSize = m_RowSize;
						dataPtr[j].rowSize = m_RowSize;
						dataPtr[j].minLocation = minPos;
						minPos += nRuns;
						dataPtr[j].maxLocation = minPos;
						mainPromise[j].set_value(&(dataPtr[j]));
					}
				}
				for (j = minPos; j < m_RowSize; j++)
				{
					pHouse->rightMultiply(m_RightTransMatrix, m_RowSize, m_RowSize, j);
				}
				if (transThread != 0 && m_RowSize >= threadLimit * transThread)
				{
					for (j = matrixThread; j < m_Thread; j++)
					{
						std::future<void> threadFuture = threadPromise[j].get_future();
						threadFuture.get();
						std::promise<void> temp;
						threadPromise[j] = std::move(temp);
					}
				}
			}
			if (matrixThread != 0 && (nEnd - nStart) >= threadLimit * matrixThread)
			{
				for (j = 0; j < matrixThread; j++)
				{
					std::future<void> threadFuture = threadPromise[j].get_future();
					threadFuture.get();
					std::promise<void> temp;
					threadPromise[j] = std::move(temp);
				}
			}
			delete[] dataPtr;
		}
	}
}

// Multiply all required columns or rows by Givens matrix using multiple threads
// Only the tranformation matrices will be calculated
void GeneralMatrix::multiplyByGivens(
	std::promise<TDataPtr>* mainPromise,
	std::promise<void>* threadPromise,
	GivensPtr pGivens,
	bool byLeft)
{
	MInt j;
	MInt minPos;
	MInt nRuns;
	if (byLeft)
	{
		if (m_Thread == 0)
		{
			for (j = 0; j < m_ColSize; j++)
			{
				pGivens->leftMultiply(m_LeftTransMatrix, m_ColSize, m_ColSize, j);
			}
		}
		else
		{
			TDataPtr dataPtr = new ThreadingDatta[m_Thread];
			minPos = 0;
			if (m_ColSize >= threadLimit * m_Thread)
			{
				nRuns = m_ColSize / m_Thread;
				for (j = 0; j < m_Thread; j++)
				{
					dataPtr[j].byHouse = false;
					dataPtr[j].gPtr = pGivens;
					dataPtr[j].hPtr = nullptr;
					dataPtr[j].dMatrix = m_LeftTransMatrix;
					dataPtr[j].colSize = m_ColSize;
					dataPtr[j].rowSize = m_ColSize;
					dataPtr[j].minLocation = minPos;
					minPos += nRuns;
					dataPtr[j].maxLocation = minPos;
					mainPromise[j].set_value(&(dataPtr[j]));
				}
			}
			for (j = minPos; j < m_ColSize; j++)
			{
				pGivens->leftMultiply(m_LeftTransMatrix, m_ColSize, m_ColSize, j);
			}
			if (m_ColSize >= threadLimit * m_Thread)
			{
				for (j = 0; j < m_Thread; j++)
				{
					std::future<void> threadFuture = threadPromise[j].get_future();
					threadFuture.get();
					std::promise<void> temp;
					threadPromise[j] = std::move(temp);
				}
			}
			delete[] dataPtr;
		}
	}
	else
	{
		if (m_Thread == 0)
		{
			for (j = 0; j < m_RowSize; j++)
			{
				pGivens->rightMultiply(m_RightTransMatrix, m_RowSize, m_RowSize, j);
			}
		}
		else
		{
			TDataPtr dataPtr = new ThreadingDatta[m_Thread];
			minPos = 0;
			if (m_RowSize >= threadLimit * m_Thread)
			{
				nRuns = m_RowSize / m_Thread;
				for (j = 0; j < m_Thread; j++)
				{
					dataPtr[j].byHouse = false;
					dataPtr[j].gPtr = pGivens;
					dataPtr[j].hPtr = nullptr;
					dataPtr[j].dMatrix = m_RightTransMatrix;
					dataPtr[j].colSize = m_RowSize;
					dataPtr[j].rowSize = m_RowSize;
					dataPtr[j].minLocation = minPos;
					minPos += nRuns;
					dataPtr[j].maxLocation = minPos;
					mainPromise[j].set_value(&(dataPtr[j]));
				}
			}
			for (j = minPos; j < m_RowSize; j++)
			{
				pGivens->rightMultiply(m_RightTransMatrix, m_RowSize, m_RowSize, j);
			}
			if (m_RowSize >= threadLimit * m_Thread)
			{
				for (j = 0; j < m_Thread; j++)
				{
					std::future<void> threadFuture = threadPromise[j].get_future();
					threadFuture.get();
					std::promise<void> temp;
					threadPromise[j] = std::move(temp);
				}
			}
			delete[] dataPtr;
		}
	}
}

void GeneralMatrix::toUpperBiDiagonal()
{
	HousePtr pHouse;
	std::promise<TDataPtr>* leftMainPromise = nullptr;
	std::promise<void>* leftThreadPromise = nullptr;
	std::thread* leftThread = nullptr;
	std::promise<TDataPtr>* rightMainPromise = nullptr;
	std::promise<void>* rightThreadPromise = nullptr;
	std::thread* rightThread = nullptr;
	if (m_Thread > 0)
	{
		leftMainPromise = new std::promise<TDataPtr>[m_Thread];
		leftThreadPromise = new std::promise<void>[m_Thread];
		leftThread = new std::thread[m_Thread];
		rightMainPromise = new std::promise<TDataPtr>[m_Thread];
		rightThreadPromise = new std::promise<void>[m_Thread];
		rightThread = new std::thread[m_Thread];
		for (MInt i = 0; i < m_Thread; i++)
		{
			leftThread[i] = std::thread(leftMultiplyByTransMatrix, leftMainPromise,
				leftThreadPromise, m_Thread, i);
			rightThread[i] = std::thread(rightMultiplyByTransMatrix, rightMainPromise,
				rightThreadPromise, m_Thread, i);
		}
	}
	MInt j;
	for (MInt i = 0; i < m_RowSize - 2; i++)
	{
		pHouse = new Householder(m_Matrix, m_ColSize, m_RowSize, i, 
			m_ColSize - i, true);
		multiplyByHouseholder(leftMainPromise, leftThreadPromise, pHouse, i + 1, 
			m_RowSize, true, true);
		m_Matrix[i][i] = (-1.0L) * pHouse->getAlpha();
		for (j = i + 1; j < m_ColSize; j++)
		{
			m_Matrix[i][j] = 0.0L;
		}
		delete pHouse;

		pHouse = new Householder(m_Matrix, m_ColSize, m_RowSize, i, 
			m_RowSize - 1 - i, false);
		multiplyByHouseholder(rightMainPromise, rightThreadPromise, pHouse,
			i + 1, m_ColSize, false, true);
		m_Matrix[i + 1][i] = (-1.0L) * pHouse->getAlpha();
		for (j = i + 2; j < m_RowSize; j++)
		{
			m_Matrix[j][i] = 0.0L;
		}
		delete pHouse;
	}
	pHouse = new Householder(m_Matrix, m_ColSize, m_RowSize, m_RowSize - 2,
		m_ColSize - m_RowSize + 2, true);
	multiplyByHouseholder(leftMainPromise, leftThreadPromise, pHouse, m_RowSize - 1,
		m_RowSize, true, true);
	m_Matrix[m_RowSize - 2][m_RowSize - 2] = (-1.0L) * pHouse->getAlpha();
	for (j = m_RowSize - 1; j < m_ColSize; j++)
	{
		m_Matrix[m_RowSize - 2][j] = 0.0L;
	}
	delete pHouse;

	pHouse = new Householder(m_Matrix, m_ColSize, m_RowSize, m_RowSize - 1, 
		m_ColSize - m_RowSize + 1, true);
	multiplyByHouseholder(leftMainPromise, leftThreadPromise, pHouse, m_RowSize,
		m_RowSize, true, true);
	m_Matrix[m_RowSize - 1][m_RowSize - 1] = (-1.0L) * pHouse->getAlpha();
	for (j = m_RowSize; j < m_ColSize; j++)
	{
		m_Matrix[m_RowSize - 1][j] = 0.0L;
	}
	delete pHouse;

	ThreadingDatta hData;
	hData.dMatrix = nullptr;
	hData.byHouse = true;
	hData.gPtr = nullptr;
	hData.hPtr = nullptr;
	for (j = 0; j < m_Thread; j++)
	{
		leftMainPromise[j].set_value(&(hData));
		leftThread[j].join();
		rightMainPromise[j].set_value(&(hData));
		rightThread[j].join();
	}
	if (m_Thread > 0)
	{
		delete[] leftMainPromise;
		delete[] leftThreadPromise;
		delete[] leftThread;
		delete[] rightMainPromise;
		delete[] rightThreadPromise;
		delete[] rightThread;
	}
}

void GeneralMatrix::singularValueDecompose()
{
	toUpperBiDiagonal();
	// Multi-threading
	std::promise<TDataPtr>* leftMainPromise = nullptr;
	std::promise<void>* leftThreadPromise = nullptr;
	std::thread* leftThread = nullptr;
	std::promise<TDataPtr>* rightMainPromise = nullptr;
	std::promise<void>* rightThreadPromise = nullptr;
	std::thread* rightThread = nullptr;
	if (m_Thread > 0)
	{
		leftMainPromise = new std::promise<TDataPtr>[m_Thread];
		leftThreadPromise = new std::promise<void>[m_Thread];
		leftThread = new std::thread[m_Thread];
		rightMainPromise = new std::promise<TDataPtr>[m_Thread];
		rightThreadPromise = new std::promise<void>[m_Thread];
		rightThread = new std::thread[m_Thread];
		for (MInt i = 0; i < m_Thread; i++)
		{
			leftThread[i] = std::thread(leftMultiplyByTransMatrix, leftMainPromise,
				leftThreadPromise, m_Thread, i);
			rightThread[i] = std::thread(rightMultiplyByTransMatrix, rightMainPromise,
				rightThreadPromise, m_Thread, i);
		}
	}
	MInt nEnd = m_RowSize - 1;
	MInt nStart;
	LDouble columnSum;
	LDouble dDelta;
	LDouble dU;
	LDouble dX;
	LDouble dY;
	LDouble tempX;
	LDouble tempY;
	GivensPtr pGivens;
	if (m_RowSize <= 1)
	{
		return;
	}
	LDouble maxColumnSum = abs(m_Matrix[0][0]) + abs(m_Matrix[1][0]);
	for (MInt i = 1; i < nEnd; i++)
	{
		columnSum = abs(m_Matrix[i][i]) + abs(m_Matrix[i + 1][i]);
		if (maxColumnSum < columnSum)
		{
			maxColumnSum = columnSum;
		}
	}
	columnSum = abs(m_Matrix[nEnd][nEnd]);
	if (maxColumnSum < columnSum)
	{
		maxColumnSum = columnSum;
	}
	while (nEnd > 0)
	{
		for (MInt i = 1; i <= nEnd; i++)
		{
			// If elements are small, set them to zero
			if (abs(m_Matrix[i][i - 1]) <
				(std::max((abs(m_Matrix[i][i]) + abs(m_Matrix[i - 1][i - 1])) * NUMERIC_ERROR,
					NUMERIC_ERROR)))
			{
				m_Matrix[i][i - 1] = 0.0L;
			}
			if (abs(m_Matrix[i - 1][i - 1]) < std::max(maxColumnSum * NUMERIC_ERROR, 
				NUMERIC_ERROR))
			{
				m_Matrix[i - 1][i - 1] = 0.0L;
			}
		}
		if (abs(m_Matrix[nEnd][nEnd]) < std::max(maxColumnSum * NUMERIC_ERROR, NUMERIC_ERROR))
		{
			m_Matrix[nEnd][nEnd] = 0.0L;
		}
		// Stop at non-zero position
		while (nEnd > 0 && abs(m_Matrix[nEnd][nEnd - 1]) < NUMERIC_ERROR)
		{
			nEnd--;
		}
		if (nEnd == 0)
		{
			break;
		}
		// If element at the last diagonal position is zero, remove the element above it
		// For example, will modify the following matrix
		// * * 0 0 0 0
		// 0 * * 0 0 0
		// 0 0 * * 0 0
		// 0 0 0 * * 0
		// 0 0 0 0 * *
		// 0 0 0 0 0 0
		// 0 0 0 0 0 0
		// To the following
		// * * 0 0 0 0
		// 0 * * 0 0 0
		// 0 0 * * 0 0
		// 0 0 0 * * 0
		// 0 0 0 0 * 0
		// 0 0 0 0 0 0
		// 0 0 0 0 0 0
		if (abs(m_Matrix[nEnd][nEnd]) < NUMERIC_ERROR)
		{
			for (MInt i = nEnd; i > 1; i--)
			{
				pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, i - 1, nEnd, i - 1, false);
				multiplyByGivens(rightMainPromise, rightThreadPromise, pGivens, false);
				pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, i - 1);
				pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, i - 2);
				delete pGivens;
			}
			pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, 0, nEnd, 0, false);
			multiplyByGivens(rightMainPromise, rightThreadPromise, pGivens, false);
			pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, 0);
			delete pGivens;
			nEnd--;
		}
		else
		{
			nStart = nEnd;
			while (nStart > 0 && (abs(m_Matrix[nStart][nStart - 1]) >= NUMERIC_ERROR
				&& abs(m_Matrix[nStart - 1][nStart - 1]) >= NUMERIC_ERROR))
			{
				nStart--;
			}
			// If the first diagonal element is zero, change it to zero.  For example,change matrix
			// 0 * 0 0 0 0
			// 0 * * 0 0 0
			// 0 0 * * 0 0
			// 0 0 0 * * 0
			// 0 0 0 0 * *
			// 0 0 0 0 0 *
			// 0 0 0 0 0 0
			// To the following
			// 0 0 0 0 0 0
			// 0 * * 0 0 0
			// 0 0 * * 0 0
			// 0 0 0 * * 0
			// 0 0 0 0 * *
			// 0 0 0 0 0 *
			// 0 0 0 0 0 0
			if (nStart > 0 && abs(m_Matrix[nStart - 1][nStart - 1]) < NUMERIC_ERROR)
			{
				for (MInt i = nStart; i < nEnd; i++)
				{
					pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, i, nStart - 1, i, true);
					multiplyByGivens(leftMainPromise, leftThreadPromise, pGivens, true);
					pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, i);
					pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, i + 1);
					delete pGivens;
				}
				pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, nEnd, nStart - 1, nEnd, true);
				multiplyByGivens(leftMainPromise, leftThreadPromise, pGivens, true);
				pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, nEnd);
				delete pGivens;
			}
			// nStart != nEnd
			else
			{
				dX = m_Matrix[nEnd][nEnd] * m_Matrix[nEnd][nEnd]
					+ m_Matrix[nEnd][nEnd - 1] * m_Matrix[nEnd][nEnd - 1];
				dY = m_Matrix[nEnd - 1][nEnd - 1] * m_Matrix[nEnd][nEnd - 1];
				if (nEnd > 1)
				{
					dDelta = (m_Matrix[nEnd - 1][nEnd - 1] * m_Matrix[nEnd - 1][nEnd - 1]
						+ m_Matrix[nEnd - 1][nEnd - 2] * m_Matrix[nEnd - 1][nEnd - 2] - dX) / 2.0L;
				}
				else
				{
					dDelta = (m_Matrix[nEnd - 1][nEnd - 1] * m_Matrix[nEnd - 1][nEnd - 1] - dX) / 2.0L;
				}
				if (abs(dDelta) < NUMERIC_ERROR)
				{
					dU = dX - dY;
				}
				else
				{
					dU = dX - dY * dY / (dDelta + dDelta / abs(dDelta) * sqrtl(dDelta * dDelta + dY * dY));
				}
				dX = m_Matrix[nStart][nStart] * m_Matrix[nStart][nStart] - dU;
				dY = m_Matrix[nStart][nStart] * m_Matrix[nStart + 1][nStart];
				tempX = m_Matrix[nStart][nStart];
				m_Matrix[nStart][nStart] = dX;
				tempY = m_Matrix[nStart + 1][nStart];
				m_Matrix[nStart + 1][nStart] = dY;
				pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, nStart, nStart + 1, nStart, false);
				m_Matrix[nStart][nStart] = tempX;
				m_Matrix[nStart + 1][nStart] = tempY;
				multiplyByGivens(rightMainPromise, rightThreadPromise, pGivens, false);
				pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, nStart);
				pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, nStart + 1);
				delete pGivens;
				for (MInt i = nStart; i < nEnd - 1; i++)
				{
					pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, i, i + 1, i, true);
					multiplyByGivens(leftMainPromise, leftThreadPromise, pGivens, true);
					pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, i);
					pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, i + 1);
					pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, i + 2);
					delete pGivens;
					pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, i, i + 2, i + 1, false);
					multiplyByGivens(rightMainPromise, rightThreadPromise, pGivens, false);
					pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, i);
					pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, i + 1);
					pGivens->rightMultiply(m_Matrix, m_ColSize, m_RowSize, i + 2);
					delete pGivens;
				}
				pGivens = new Givens(m_Matrix, m_ColSize, m_RowSize, nEnd - 1, nEnd, nEnd - 1, true);
				multiplyByGivens(leftMainPromise, leftThreadPromise, pGivens, true);
				pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, nEnd - 1);
				pGivens->leftMultiply(m_Matrix, m_ColSize, m_RowSize, nEnd);
				delete pGivens;
			}
		}
	}
	if (m_Thread > 0)
	{
		ThreadingDatta hData;
		hData.dMatrix = nullptr;
		hData.byHouse = true;
		hData.gPtr = nullptr;
		hData.hPtr = nullptr;
		for (MInt i = 0; i < m_Thread; i++)
		{
			leftMainPromise[i].set_value(&(hData));
			leftThread[i].join();
			rightMainPromise[i].set_value(&(hData));
			rightThread[i].join();
		}
		if (leftMainPromise)
		{
			delete[] leftMainPromise;
		}
		if (leftThreadPromise)
		{
			delete[] leftThreadPromise;
		}
		if (leftThread)
		{
			delete[] leftThread;
		}
		if (rightMainPromise)
		{
			delete[] rightMainPromise;
		}
		if (rightThreadPromise)
		{
			delete[] rightThreadPromise;
		}
		if (rightThread)
		{
			delete[] rightThread;
		}
	}
}

void GeneralMatrix::setValue(MInt colPos, MInt rowPos, LDouble dValue)
{
	if (colPos >= m_RowSize || rowPos >= m_ColSize)
	{
		throw std::invalid_argument("Position cannot be larger than size.");
	}
	m_Matrix[colPos][rowPos] = dValue;
}

LDouble GeneralMatrix::getValue(MInt colPos, MInt rowPos)
{
	if (colPos >= m_RowSize || rowPos >= m_ColSize)
	{
		throw std::invalid_argument("Position cannot be larger than size.");
	}
	return m_Matrix[colPos][rowPos];
}
