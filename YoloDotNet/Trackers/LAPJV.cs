// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// LAPJV（使用 Jonker-Volgenant 算法的线性分配问题）求解器，用于矩形成本矩阵。
    /// 找到以最低总成本匹配两组项目的最佳方式。
    /// "成本"表示匹配的糟糕程度；成本越低意味着匹配越好。
    /// </summary>
    public static class LAPJV
    {
        // 最小容差值，用于判断什么算作零，以处理检查零约简成本时的微小浮点误差
        //（用于识别候选分配）。
        private const float BASE_EPSILON = 1e-6f;

        private const float PAD_MULTIPLIER = 10f;
        private const float PAD_OFFSET = 1f;

        /// <summary>
        /// 求解线性分配问题以最小化总成本。
        /// </summary>
        /// <param name="costMatrix">输入成本矩阵（行 x 列）。</param>
        /// <returns>
        /// 分配数组：每行分配的列索引（如果未分配则为 -1）。
        /// 返回数组的长度等于输入行数。
        /// </returns>
        public static int[] Solve(float[,] costMatrix)
        {
            if (costMatrix == null)
                throw new YoloDotNetException(nameof(costMatrix));

            int nRows = costMatrix.GetLength(0);
            int nCols = costMatrix.GetLength(1);

            if (nRows == 0 || nCols == 0)
                throw new YoloDotNetException("Cost matrix must have non-zero dimensions.");

            int n = Math.Max(nRows, nCols);
            float[,] cost = PadCostMatrix(costMatrix, nRows, nCols, n, out float maxCost);
            float epsilon = Math.Max(BASE_EPSILON, maxCost * 1e-6f);

            float[] u = new float[n];
            float[] v = new float[n];

            int[] rowsol = CreateInitializedArray(n, -1);
            int[] colsol = CreateInitializedArray(n, -1);

            ReduceRows(cost, u, n);
            ReduceColumns(cost, v, n);

            InitialAssignment(cost, rowsol, colsol, epsilon, n);

            Augment(cost, u, v, rowsol, colsol, n, epsilon);

            return ExtractAssignment(rowsol, nRows, nCols);
        }

        /// <summary>
        /// 通过添加填充大量惩罚值的额外行或列，将成本矩阵填充为方阵（行数和列数相同）。
        /// 这确保了即使两组大小不同，分配算法也能正常工作。
        /// </summary>
        private static float[,] PadCostMatrix(float[,] costMatrix, int nRows, int nCols, int n, out float maxCost)
        {
            // 创建一个新的 n x n 方阵（n = 行数和列数的最大值）
            float[,] cost = new float[n, n];

            // 将 maxCost 初始化为可能的最小浮点值
            maxCost = float.MinValue;

            // 将原始成本矩阵复制到新的方阵中
            // 同时查找最大成本值以供后续使用
            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    float val = costMatrix[i, j];
                    cost[i, j] = val;

                    if (val > maxCost)
                        maxCost = val; // 跟踪找到的最高成本值
                }
            }

            // 计算分配给填充单元格（虚拟行/列）的大惩罚成本 (bigM)
            // 此值设置得高于任何实际成本，以阻止与虚拟条目的匹配
            float bigM = (maxCost * PAD_MULTIPLIER) + PAD_OFFSET;

            // 用 bigM 惩罚成本填充底部行（虚拟行）
            for (int i = nRows; i < n; i++)
                for (int j = 0; j < n; j++)
                    cost[i, j] = bigM;

            // 用 bigM 惩罚成本填充最右列（虚拟列）
            for (int i = 0; i < n; i++)
                for (int j = nCols; j < n; j++)
                    cost[i, j] = bigM;

            // 返回填充了大惩罚成本的新方阵成本矩阵
            return cost;
        }

        /// <summary>
        /// Creates an integer array of the given length where all elements are set to the specified initial value.
        /// Used to set up assignment arrays with default values (e.g. -1 means unassigned).
        /// </summary>
        private static int[] CreateInitializedArray(int length, int value)
        {
            // Create a new integer array of the requested length
            int[] array = new int[length];

            // Fill every element in the array with the given value
            // This is useful for initializing arrays to a default (like -1 for unassigned)
            for (int i = 0; i < length; i++)
                array[i] = value;

            // Return the initialized array
            return array;
        }

        /// <summary>
        /// Subtracts the smallest value in each row from all elements in that row,
        /// making sure each row has at least one zero-cost position.
        /// Updates the dual variable for rows (u) with these minimum values.
        /// This helps prepare the cost matrix for finding optimal assignments.
        /// </summary>
        private static void ReduceRows(float[,] cost, float[] u, int n)
        {
            // Iterate through each row in the square cost matrix
            for (int i = 0; i < n; i++)
            {
                // Find the minimum cost value in the current row
                float minVal = float.MaxValue;
                for (int j = 0; j < n; j++)
                    minVal = Math.Min(minVal, cost[i, j]);

                // Store this minimum value in the dual variable u for the row
                // This represents the smallest cost we can subtract from the row without making any value negative
                u[i] = minVal;

                // Subtract the minimum value from every element in the row
                // This step ensures at least one zero in each row, which helps in finding assignments
                for (int j = 0; j < n; j++)
                    cost[i, j] -= minVal;
            }
        }

        /// <summary>
        /// Subtracts the smallest value in each column from all elements in that column,
        /// making sure each column has at least one zero-cost position.
        /// Updates the dual variable for columns (v) with these minimum values.
        /// This helps prepare the cost matrix for finding optimal assignments.
        /// </summary>
        private static void ReduceColumns(float[,] cost, float[] v, int n)
        {
            // Iterate through each column in the square cost matrix
            for (int j = 0; j < n; j++)
            {
                // Find the minimum cost value in the current column
                float minVal = float.MaxValue;
                for (int i = 0; i < n; i++)
                    minVal = Math.Min(minVal, cost[i, j]);

                // Store this minimum value in the dual variable v for the column
                // This represents the smallest cost we can subtract from the column without making any value negative
                v[j] = minVal;

                // Subtract the minimum value from every element in the column
                // This ensures at least one zero in each column, aiding in the assignment process
                for (int i = 0; i < n; i++)
                    cost[i, j] -= minVal;
            }
        }

        /// <summary>
        /// Makes initial assignments by matching rows to columns where the cost is effectively zero (within a small tolerance).
        /// A column is assigned to a row only if it is not already taken.
        /// This sets up the starting solution for further refinement.
        /// </summary>
        private static void InitialAssignment(float[,] cost, int[] rowsol, int[] colsol, float epsilon, int n)
        {
            // For each row in the cost matrix
            for (int i = 0; i < n; i++)
            {
                // For each column in the cost matrix
                for (int j = 0; j < n; j++)
                {
                    // Check two conditions:
                    // 1. The column is not already assigned (colsol[j] == -1)
                    // 2. The cost value is close enough to zero (within epsilon tolerance)
                    //    This means this pair (row i, column j) is a promising candidate for assignment.
                    if (colsol[j] == -1 && Math.Abs(cost[i, j]) < epsilon)
                    {
                        // Assign column j to row i
                        rowsol[i] = j;

                        // Mark column j as assigned to row i
                        colsol[j] = i;

                        // Move to the next row since this row now has an assignment
                        break;
                    }
                }
            }
        }

        /// <summary>
        /// Finds and builds augmenting paths to complete the assignment so that every row is matched to a column.
        /// It searches for the best way to connect free rows to free columns, updating the solution
        /// while keeping the total cost as low as possible.
        /// Adjusts the dual variables (u, v) during the process to reflect cost changes.
        /// </summary>
        private static void Augment(float[,] cost, float[] u, float[] v, int[] rowsol, int[] colsol, int n, float epsilon)
        {
            // Distance array: stores shortest distances from the free row to columns
            float[] dist = new float[n];

            // Predecessor array: to reconstruct the augmenting path
            int[] pred = new int[n];

            // Seen array: tracks which columns have been visited in the search
            bool[] seen = new bool[n];

            // Iterate over all rows to find free rows (not yet assigned)
            for (int freeRow = 0; freeRow < n; freeRow++)
            {
                // Skip if this row is already assigned to a column
                if (rowsol[freeRow] != -1)
                    continue;

                // Initialize the distances, predecessors, and seen flags for this free row
                for (int j = 0; j < n; j++)
                {
                    // Reduced cost for assigning freeRow to column j, adjusted by dual variables
                    dist[j] = cost[freeRow, j] - u[freeRow] - v[j];

                    // Set the predecessor for each column as the freeRow initially
                    pred[j] = freeRow;

                    // None of the columns have been visited yet
                    seen[j] = false;
                }

                int j0 = -1;  // Will store the column index where augmenting path ends

                while (true)
                {
                    // Find the unseen column with the minimum distance (best candidate for augmenting path)
                    int j = FindMinUnseenColumn(dist, seen, n, out float minDist);

                    // If no such column is found, it means no augmenting path exists => throw error
                    if (j == -1)
                        throw new YoloDotNetException(
                            $"No augmenting path found for free row {freeRow}. Verify cost matrix contains valid, finite costs.");

                    // Mark this column as seen (visited)
                    seen[j] = true;

                    // If this column is free (not assigned to any row), we found the end of augmenting path
                    if (colsol[j] == -1)
                    {
                        j0 = j;
                        break;
                    }

                    // Otherwise, explore the row currently assigned to this column
                    int i = colsol[j];
                    float distJ = dist[j];

                    // Update distances for columns not yet seen based on new candidate path through row i
                    for (int k = 0; k < n; k++)
                    {
                        if (seen[k])
                            continue;

                        // Calculate new tentative distance from freeRow to column k via row i and column j
                        float newDist = cost[i, k] - u[i] - v[k] + distJ;

                        // If this new distance is shorter, update it and set predecessor accordingly
                        if (newDist < dist[k])
                        {
                            dist[k] = newDist;
                            pred[k] = i;
                        }
                    }
                }

                // After finding the augmenting path, update the dual variables and assignments
                UpdateDualsAndAugment(dist, seen, u, v, rowsol, colsol, pred, freeRow, j0, n);
            }
        }

        /// <summary>
        /// Finds the column that hasn’t been checked yet (unseen) and has the smallest distance value.
        /// This helps choose the next best column to explore when building an augmenting path.
        /// </summary>
        /// <param name="dist">Array of distances to columns.</param>
        /// <param name="seen">Array marking which columns have been visited.</param>
        /// <param name="n">Number of columns (size).</param>
        /// <param name="minDist">Output parameter for the minimum distance found.</param>
        /// <returns>The index of the unseen column with the smallest distance, or -1 if none found.</returns>
        private static int FindMinUnseenColumn(float[] dist, bool[] seen, int n, out float minDist)
        {
            // Initialize minimum distance with the largest possible float value
            minDist = float.MaxValue;
            int minJ = -1;  // Index of the column with the smallest distance found

            // Iterate over all columns
            for (int j = 0; j < n; j++)
            {
                // Check if column j is unseen and if its distance is less than the current minimum
                if (!seen[j] && dist[j] < minDist)
                {
                    // Update minDist and remember the column index
                    minDist = dist[j];
                    minJ = j;
                }
            }

            // Return the column index with the smallest distance among unseen columns, or -1 if none
            return minJ;
        }

        /// <summary>
        /// Updates the dual variables (used for adjusting costs) and applies the new assignments found by the augmenting path.
        /// This step finalizes the improvements to the matching.
        /// </summary>
        /// <param name="dist">Array of distances to columns.</param>
        /// <param name="seen">Array marking which columns are part of the alternating tree.</param>
        /// <param name="u">Dual variables for rows.</param>
        /// <param name="v">Dual variables for columns.</param>
        /// <param name="rowsol">Current row-to-column assignments.</param>
        /// <param name="colsol">Current column-to-row assignments.</param>
        /// <param name="pred">Predecessor array used to reconstruct the augmenting path.</param>
        /// <param name="freeRow">The free row to be assigned.</param>
        /// <param name="j0">The ending column of the augmenting path.</param>
        /// <param name="n">The size of the problem (number of rows/columns).</param>
        private static void UpdateDualsAndAugment(
            float[] dist,
            bool[] seen,
            float[] u,
            float[] v,
            int[] rowsol,
            int[] colsol,
            int[] pred,
            int freeRow,
            int j0,
            int n)
        {
            // The smallest distance to the column j0 on the augmenting path
            float delta = dist[j0];

            // Update the dual variables for columns in the alternating tree
            for (int j = 0; j < n; j++)
            {
                if (seen[j])
                {
                    // Columns that are part of the tree get their dual variable increased by delta
                    v[j] += delta;
                }
                else
                {
                    // Columns not in the tree reduce their distance values by delta to maintain feasibility
                    dist[j] -= delta;
                }
            }

            // Increase the dual variable for the free row by delta to keep feasibility
            u[freeRow] += delta;

            // Start from the last column in the augmenting path
            int curJ = j0;

            // Follow the predecessors backward to update the assignment arrays
            while (true)
            {
                int curI = pred[curJ];   // Row associated with the current column in the path
                int nextJ = rowsol[curI]; // The previously assigned column for this row

                // Assign current column to the row in rowsol and update colsol accordingly
                rowsol[curI] = curJ;
                colsol[curJ] = curI;

                // Stop once we reach the free row where the path started
                if (curI == freeRow)
                    break;

                // Move backward along the augmenting path
                curJ = nextJ;
            }
        }

        /// <summary>
        /// Builds the final list of assignments by selecting only the real columns (ignores dummy columns added for padding).
        /// Returns -1 if no valid assignment was made for a row.
        /// </summary>
        /// <param name="rowsol">Array where each element gives the assigned column for the corresponding row (including padded columns).</param>
        /// <param name="nRows">Original number of rows (before padding).</param>
        /// <param name="nCols">Original number of columns (before padding).</param>
        /// <returns>
        /// An array where each element is the assigned column index for that row. 
        /// If a row is assigned to a dummy column (outside original columns), it returns -1 indicating no valid assignment.
        private static int[] ExtractAssignment(int[] rowsol, int nRows, int nCols)
        {
            // Create result array sized to the original number of rows
            int[] assignment = new int[nRows];

            // Iterate over each original row
            for (int i = 0; i < nRows; i++)
            {
                int col = rowsol[i]; // Assigned column index for this row, possibly padded

                // If the assignment is to a real column (within original column count), keep it
                // Otherwise, assign -1 to indicate no valid assignment (dummy column)
                assignment[i] = (col < nCols) ? col : -1;
            }

            // Return the filtered assignments
            return assignment;
        }
    }
}
