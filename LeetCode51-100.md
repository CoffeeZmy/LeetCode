# LeetCode51-100

## 51.N皇后

### 题目

n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。

### 题解

```java
private List<List<String>> result = new ArrayList<>();

public List<List<String>> solveNQueens(int n) {

    char[][] board = new char[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            board[i][j] = '.';
        }
    }
    // 每行下了棋的坐标
    Map<Integer, Integer> position = new HashMap<>();
    putChess(board, 0, position);

    return result;
}

/**
 * 下棋
 */
private void putChess(char[][] board, int rowIndex, Map<Integer, Integer> position) {
    int n = board.length;
    if (rowIndex == n) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            res.add(new String(board[i]));
        }
        result.add(res);
        return;
    }

    for (int col = 0; col < n; col++) {
        if (canPut(rowIndex, col, position)) {

            board[rowIndex][col] = 'Q';
            position.put(rowIndex, col);

            // 下下一行
            putChess(board, rowIndex + 1, position);

            board[rowIndex][col] = '.';
            position.remove(rowIndex);
        }
    }
}

/**
 * 判断是否能下
 */
private boolean canPut(int row, int col, Map<Integer, Integer> position) {
    for (int i = 0; i < row; i++) {
        int j = position.get(i);
        if (j == col || (row - i == Math.abs(j - col))) {
            return false;
        }
    }
    return true;
}
```

## 52.N皇后2

### 题目

同51，求解法数量

### 题解

#### 同51

```java
private int result = 0;

public int totalNQueens(int n) {

    char[][] board = new char[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            board[i][j] = '.';
        }
    }
    // 每行下了棋的坐标
    Map<Integer, Integer> position = new HashMap<>();
    putChess(board, 0, position);

    return result;
}

/**
 * 下棋
 */
private void putChess(char[][] board, int rowIndex, Map<Integer, Integer> position) {
    int n = board.length;
    if (rowIndex == n) {
        result++;
        return;
    }

    for (int col = 0; col < n; col++) {
        if (canPut(rowIndex, col, position)) {

            board[rowIndex][col] = 'Q';
            position.put(rowIndex, col);

            // 下下一行
            putChess(board, rowIndex + 1, position);

            board[rowIndex][col] = '.';
            position.remove(rowIndex);
        }
    }
}

/**
 * 判断是否能下
 */
private boolean canPut(int row, int col, Map<Integer, Integer> position) {
    for (int i = 0; i < row; i++) {
        int j = position.get(i);
        if (j == col || (row - i == Math.abs(j - col))) {
            return false;
        }
    }
    return true;
}
```

#### 去掉棋盘（减少内存使用）

```java
private int result = 0;

public int totalNQueens(int n) {
    // 每行下了棋的坐标
    Map<Integer, Integer> position = new HashMap<>();
    putChess(0, position, n);
    return result;
}

/**
 * 下棋
 */
private void putChess(int rowIndex, Map<Integer, Integer> position, int n) {
    if (rowIndex == n) {
        result++;
        return;
    }

    for (int col = 0; col < n; col++) {
        if (canPut(rowIndex, col, position)) {
            position.put(rowIndex, col);
            // 下下一行
            putChess(rowIndex + 1, position, n);
            position.remove(rowIndex);
        }
    }
}

/**
 * 判断是否能下
 */
private boolean canPut(int row, int col, Map<Integer, Integer> position) {
    for (int i = 0; i < row; i++) {
        int j = position.get(i);
        if (j == col || (row - i == Math.abs(j - col))) {
            return false;
        }
    }
    return true;
}
```

## 53.最大子序和

### 题目

给定一个整数数组nums，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和

### 题解

#### O(n)解法

```java
public int maxSubArray(int[] nums) {
    int res = nums[0];

    int i = 1, temp = res;
    while (i < nums.length) {
        if (temp <= 0) {
            temp = nums[i];
        } else {
            temp += nums[i];
        }
        i++;
        if (temp > res) {
            res = temp;
        }
    }
    return res;
}
```

#### 分治法（看看就好）

```java
public class Status {
    public int lSum, rSum, mSum, iSum;

    public Status(int lSum, int rSum, int mSum, int iSum) {
        this.lSum = lSum;
        this.rSum = rSum;
        this.mSum = mSum;
        this.iSum = iSum;
    }
}

public int maxSubArray(int[] nums) {
    return getInfo(nums, 0, nums.length - 1).mSum;
}

public Status getInfo(int[] a, int l, int r) {
    if (l == r) {
        return new Status(a[l], a[l], a[l], a[l]);
    }
    int m = (l + r) >> 1;
    Status lSub = getInfo(a, l, m);
    Status rSub = getInfo(a, m + 1, r);
    return pushUp(lSub, rSub);
}

public Status pushUp(Status l, Status r) {
    int iSum = l.iSum + r.iSum;
    int lSum = Math.max(l.lSum, l.iSum + r.lSum);
    int rSum = Math.max(r.rSum, r.iSum + l.rSum);
    int mSum = Math.max(Math.max(l.mSum, r.mSum), l.rSum + r.lSum);
    return new Status(lSum, rSum, mSum, iSum);
}
```

## 54.螺旋矩阵

### 题目

给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

### 题解

#### 状态机

```java
private enum Status {
    ToRight,
    Down,
    ToLeft,
    Up
}

public List<Integer> spiralOrder(int[][] matrix) {
    List<Integer> result = new ArrayList<>();
    int rowLen = matrix.length;

    if (rowLen == 0) {
        return result;
    }

    int colLen = matrix[0].length;

    int rowMin = 0, rowMax = rowLen - 1, colMin = 0, colMax = colLen - 1;
    Status curStatus = Status.ToRight;

    int i = 0, j = 0, count = rowLen * colLen;
    while (count > 0) {
        result.add(matrix[i][j]);
        count--;

        switch (curStatus) {
            case ToRight:
                if (j == colMax) {
                    rowMin++;
                    curStatus = Status.Down;
                    i++;
                } else {
                    j++;
                }
                break;
            case Down:
                if (i == rowMax) {
                    colMax--;
                    curStatus = Status.ToLeft;
                    j--;
                } else {
                    i++;
                }
                break;
            case ToLeft:
                if (j == colMin) {
                    rowMax--;
                    curStatus = Status.Up;
                    i--;
                } else {
                    j--;
                }
                break;
            case Up:
                if (i == rowMin) {
                    colMin++;
                    curStatus = Status.ToRight;
                    j++;
                } else {
                    i--;
                }
                break;
        }
    }
    return result;
}
```

## 55.跳跃游戏

### 题目

给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个位置。

### 题解

#### dp

```java
int len = nums.length;
boolean[] dp = new boolean[len];

dp[len - 1] = true;

int i = len - 2;
while (i >= 0) {
    if (nums[i] >= len - i - 1) {
        dp[i--] = true;
        continue;
    }
    int temp = i + 1;
    boolean flag = false;
    while (temp < len && temp <= i + nums[i]) {
        if (dp[temp]) {
            dp[i--] = true;
            flag = true;
            break;
        }
        temp++;
    }
    if (!flag) {
        dp[i--] = false;
    }
}		
return dp[0];
```

#### 贪心算法

```java
public boolean canJump(int[] nums) {
    int n = nums.length;
    // 能到达的最右的距离
    int rightmost = 0;
    for (int i = 0; i < n; ++i) {
        if (i <= rightmost) {
            rightmost = Math.max(rightmost, i + nums[i]);
            if (rightmost >= n - 1) {
                return true;
            }
        }
    }
    return false;
}
```

## 56.合并区间

### 题目

给出一个区间的集合，请合并所有重叠的区间

### 题解

#### 冒泡排序

```java
public int[][] merge(int[][] intervals) {
    int len = intervals.length;

    if (len == 0 || len == 1) {
        return intervals;
    }

    // 冒泡排序
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - 1 - i; j++) {
            if (intervals[j][0] > intervals[j + 1][0]) {
                int[] temp = intervals[j];
                intervals[j] = intervals[j + 1];
                intervals[j + 1] = temp;
            }
        }
    }
    int[][] result = new int[len][2];
    int[] mergeTemp = intervals[0];
    int i = 1, index = 0;
    while (i < len) {
        if (mergeTemp[0] <= intervals[i][0] && mergeTemp[1] >= intervals[i][0]) {
            mergeTemp[1] = Math.max(mergeTemp[1], intervals[i][1]);
        } else {
            result[index] = mergeTemp;
            index++;
            mergeTemp = intervals[i];
        }
        i++;

        if (i == len) {
            result[index] = mergeTemp;
        }
    }
    return Arrays.copyOf(result, index + 1);
}
```

#### 不要自己写排序

```java
public int[][] merge(int[][] intervals) {
    int len = intervals.length;

    if (len == 0 || len == 1) {
        return intervals;
    }

    Arrays.sort(intervals, (int[] a, int[] b) -> a[0] - b[0]);

    int[][] result = new int[len][2];
    int[] mergeTemp = intervals[0];
    int i = 1, index = 0;
    while (i < len) {
        if (mergeTemp[0] <= intervals[i][0] && mergeTemp[1] >= intervals[i][0]) {
            mergeTemp[1] = Math.max(mergeTemp[1], intervals[i][1]);
        } else {
            result[index] = mergeTemp;
            index++;
            mergeTemp = intervals[i];
        }
        i++;

        if (i == len) {
            result[index] = mergeTemp;
        }
    }
    return Arrays.copyOf(result, index + 1);
}
```

## 57.插入区间

### 题目

给出一个无重叠的 ，按照区间起始端点排序的区间列表。在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）

### 题解

#### 二分查找插入然后合并

```java
public int[][] insert(int[][] intervals, int[] newInterval) {
    int len = intervals.length;

    if (intervals.length == 0) {
        int[][] result = new int[1][2];
        result[0] = newInterval;
        return result;
    }

    int left = 0, right = len - 1;
    int mid;
    int insertIndex = -1;
    while (left < right) {
        mid = (left + right) / 2;

        int temp = intervals[mid][0];
        if (temp == newInterval[0]) {
            insertIndex = mid + 1;
            break;
        } else if (temp < newInterval[0]) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (insertIndex == -1) {
        if (intervals[left][0] > newInterval[0]) {
            insertIndex = left;
        } else {
            insertIndex = left + 1;
        }
    }
    int[][] tempResult = insertArray(intervals, newInterval, insertIndex);

    int start = insertIndex - 1;
    if (start < 0) {
        start = 0;
    }

    int i = start + 1;
    int count = 0;
    while (i < tempResult.length) {
        if (tempResult[start][1] >= tempResult[i][0]) {
            tempResult[start][1] = Math.max(tempResult[start][1], tempResult[i][1]);
            count++;
        } else {
            if (i >= insertIndex + 1) {
                break;
            }
            start++;
        }
        i++;
    }

    int[][] result = new int[tempResult.length - count][2];

    System.arraycopy(tempResult, 0, result, 0, start + 1);
    System.arraycopy(tempResult, start + count + 1, result, start + 1, tempResult.length - start - count - 1);
    return result;
}

private int[][] insertArray(int[][] intervals, int[] newInterval, int index) {
    int[][] result = new int[intervals.length + 1][2];
    System.arraycopy(intervals, 0, result, 0, index);
    result[index] = newInterval;
    System.arraycopy(intervals, index, result, index + 1, intervals.length - index);
    return result;
}
```

## 58.最后一个单词的长度

### 题目

给定一个仅包含大小写字母和空格 ' ' 的字符串 s，返回其最后一个单词的长度。如果字符串从左向右滚动显示，那么最后一个单词就是最后出现的单词。如果不存在最后一个单词，请返回0

### 题解

```java
public int lengthOfLastWord(String s) {
    char[] array = s.toCharArray();
    int len = 0;
    for (int i = array.length - 1; i >= 0; i--) {
        char cur = array[i];

        if (cur == ' ') {
            if (len != 0) {
                break;
            }
        } else {
            len++;
        }
    }
    return len;
}
```

## 59.螺旋矩阵2

### 题目

给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。

### 题解

```java
private enum State {
    ToRight, Down, ToLeft, Up
}

public int[][] generateMatrix(int n) {
    int total = n * n;
    int[][] result = new int[n][n];
    int i = 0, j = 0;
    int cur = 1;

    int maxRow = n - 1, minRow = 0, maxCol = n - 1, minCol = 0;

    State curState = State.ToRight;
    while (cur <= total) {
        result[i][j] = cur;

        switch (curState) {
            case ToRight:
                if (j == maxCol) {
                    i++;
                    curState = State.Down;
                    minRow++;
                } else {
                    j++;
                }
                break;
            case Down:
                if (i == maxRow) {
                    j--;
                    curState = State.ToLeft;
                    maxCol--;
                } else {
                    i++;
                }
                break;
            case ToLeft:
                if (j == minCol) {
                    i--;
                    curState = State.Up;
                    maxRow--;
                } else {
                    j--;
                }
                break;
            case Up:
                if (i == minRow) {
                    j++;
                    curState = State.ToRight;
                    minCol++;
                } else {
                    i--;
                }
                break;
        }
        cur++;
    }
    return result;
}
```

## 60.第K个排列

### 题目

给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。给定 *n* 和 *k*，返回第 *k* 个排列。

### 题解

#### 找下一个排列

```java
public String getPermutation(int n, int k) {
    int[] nums = new int[n];
    for (int i = 1; i <= n; i++) {
        nums[i - 1] = i;
    }

    int total = 1;
    int temp = 1;

    while (temp <= n) {
        total *= temp;

        if (total >= k) {
            break;
        }
        temp++;
    }
    // 第t!个序列一定是末尾t个数转置的序列
    reverse(nums, nums.length - (temp - 1), nums.length - 1);

    // 从第t!个序列到第k个还需要计算next的次数
    int nextTimes = k - total / temp;
    while (nextTimes > 0) {
        nums = getNextPermutation(nums);
        nextTimes--;
    }
    StringBuilder stringBuilder = new StringBuilder();
    for (int num : nums) {
        stringBuilder.append(num);
    }
    return stringBuilder.toString();
}

/**
 * 找下个排列
 */
public int[] getNextPermutation(int[] nums) {
    int diffPoint = -1;
    int i = nums.length - 1;

    while (i >= 1) {
        if (nums[i - 1] < nums[i]) {
            diffPoint = i - 1;
            break;
        }
        i--;
    }
    if (diffPoint == -1) {
        reverse(nums, 0, nums.length - 1);
        return nums;
    }

    int j = nums.length - 1;
    while (j > diffPoint) {
        if (nums[j] > nums[diffPoint]) {
            break;
        }
        j--;
    }
    int temp = nums[j];
    nums[j] = nums[diffPoint];
    nums[diffPoint] = temp;
    reverse(nums, diffPoint + 1, nums.length - 1);
    return nums;
}

private void reverse(int[] nums, int start, int end) {
    if (start == end) {
        return;
    }
    while (start < end) {
        int temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;

        start++;
        end--;
    }
}
```

#### 官方解（懒得看了。。）

```java
public String getPermutation(int n, int k) {
    int[] factorial = new int[n];
    factorial[0] = 1;
    for (int i = 1; i < n; ++i) {
        factorial[i] = factorial[i - 1] * i;
    }

    --k;
    StringBuffer ans = new StringBuffer();
    int[] valid = new int[n + 1];
    Arrays.fill(valid, 1);
    for (int i = 1; i <= n; ++i) {
        int order = k / factorial[n - i] + 1;
        for (int j = 1; j <= n; ++j) {
            order -= valid[j];
            if (order == 0) {
                ans.append(j);
                valid[j] = 0;
                break;
            }
        }
        k %= factorial[n - i];
    }
    return ans.toString();
}
```

## 61.旋转链表

### 题目

给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。

### 题解

```java
public ListNode rotateRight(ListNode head, int k) {
    if (head == null || k == 0 || head.next == null) {
        return head;
    }
    int temp = k - 1;
    ListNode frontNode = head;
    while (temp > 0) {
        frontNode = frontNode.next;
        temp--;

        if (frontNode.next == null) {
            return rotateRight(head, k % (k - temp));
        }
    }
    ListNode behindNode = head;

    while (true) {
        frontNode = frontNode.next;

        if (frontNode.next == null) {
            break;
        }
        behindNode = behindNode.next;
    }
    ListNode result = behindNode.next;
    behindNode.next = null;
    frontNode.next = head;
    return result;
}
```

## 62.不同路径

### 题目

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。问总共有多少条不同的路径？

### 题解

#### 数学排列问题（DP计算）

```java
public int uniquePaths(int m, int n) {
    if (m == 1 || n == 1) {
        return 1;
    }

    // 答案就是C(n-1, m + n - 2)
    // 又有C(n, m+1) = C(n , m) + C(n-1, m)

    int[][] dp = new int[m + n - 1][n];
    for (int i = 1; i < m + n - 1; i++) {
        for (int j = 1; j < n; j++) {
            if (i == 1 || i == j) {
                dp[i][j] = 1;
                continue;
            }
            if (j == 1) {
                dp[i][j] = i;
                continue;
            }
            dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }
    return dp[m + n - 2][n - 1];
}
```

## 63.不同路径2

### 题目

在62的基础上考虑路径中有障碍物障碍物在数组中记为1

### 题解

#### dp

```java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int m = obstacleGrid.length;
    int n = obstacleGrid[0].length;

    // dp[m-1][n-1]为m * n有多少种走法
    // dp[m][n] = dp[m][n - 1] + dp[m - 1][n]
    int[][] dp = new int[m][n];

    boolean hasObs = false;
    for (int j = 0; j < n; j++) {
        if (obstacleGrid[0][j] == 1) {
            hasObs = true;
        }
        dp[0][j] = hasObs ? 0 : 1;
    }
    hasObs = false;
    for (int i = 0; i < m; i++) {
        if (obstacleGrid[i][0] == 1) {
            hasObs = true;
        }
        dp[i][0] = hasObs ? 0 : 1;
    }

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[i][j] == 1) {
                dp[i][j] = 0;
            } else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }
    return dp[m - 1][n - 1];
}
```

## 64.最小路径和

### 题目

给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。每次只能向下或者向右移动一步。

### 题解

#### dp

```java
public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;

    int[][] dp = new int[m][n];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 && j == 0) {
                dp[0][0] = grid[0][0];
            } else if (i == 0) {
                dp[0][j] = dp[0][j - 1] + grid[0][j];
            } else if (j == 0) {
                dp[i][0] = dp[i - 1][0] + grid[i][0];
            } else {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
    }
    return dp[m - 1][n - 1];
}
```

## 65.有效数字

### 题目

验证给定的字符串是否可以解释为十进制数字（可包含正负号、指数e、小数点）。

### 题解

**状态机**

-1表示无效

| state                                                | 空格 | +/-  | 0-9  | .    | e    | 其他 |
| ---------------------------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0：未开始                                            | 0    | 1    | 6    | 2    | -1   | -1   |
| 1：有符号                                            | -1   | -1   | 6    | 2    | -1   | -1   |
| 2：前一位是小数点                                    | -1   | -1   | 3    | -1   | -1   | -1   |
| 3：有效未结束且前面有小数点                          | 8    | -1   | 3    | -1   | 4    | -1   |
| 4：有效未结束且前面有小数点且前一位是e               | -1   | 7    | 5    | -1   | -1   | -1   |
| 5：有效未结束且前面有小数点和e                       | 8    | -1   | 5    | -1   | -1   | -1   |
| 6：有效未结束                                        | 8    | -1   | 6    | 3    | 4    | -1   |
| 7：有效未结束且前面有小数点且前前位是e，前位是正负号 | -1   | -1   | 5    | -1   | -1   | -1   |
| 8：剩下需要全空格                                    | 8    | -1   | -1   | -1   | -1   | -1   |

```java
public boolean isNumber(String s) {
    int[][] status = {
        {0, 1, 6, 2, -1, -1},
        {-1, -1, 6, 2, -1, -1},
        {-1, -1, 3, -1, -1, -1},
        {8, -1, 3, -1, 4, -1},
        {-1, 7, 5, -1, -1, -1},
        {8, -1, 5, -1, -1, -1},
        {8, -1, 6, 3, 4, -1},
        {-1, -1, 5, -1, -1, -1},
        {8, -1, -1, -1, -1, -1}
    };

    int curStatus = 0;
    char[] array = s.toCharArray();
    boolean canBeValid = true;
    for (int i = 0; i < array.length && canBeValid; i++) {
        char cur = array[i];
        switch (cur) {
            case ' ':
                curStatus = status[curStatus][0];
                break;
            case '+':
            case '-':
                curStatus = status[curStatus][1];
                break;
            case '.':
                curStatus = status[curStatus][3];
                break;
            case 'e':
                curStatus = status[curStatus][4];
                break;
            default:
                if (cur >= '0' && cur <= '9') {
                    curStatus = status[curStatus][2];
                } else {
                    curStatus = -1;
                }
        }
        canBeValid = curStatus != -1;
    }
    if (canBeValid) {
        return curStatus == 3 || curStatus == 5 || curStatus == 6 || curStatus == 8;
    }
    return false;
}
```

## 66.加一

### 题目

给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。你可以假设除了整数 0 之外，这个整数不会以零开头

### 题解

```java
public int[] plusOne(int[] digits) {
    int carry = 1;

    int i = digits.length - 1;
    while (i >= 0) {
        if (carry == 0) {
            break;
        } else {
            int temp = digits[i];
            digits[i] = (temp + carry) % 10;
            carry = (temp + carry) / 10;
        }
        i--;
    }
    if (carry != 0) {
        int[] result = new int[digits.length + 1];
        result[0] = carry;
        System.arraycopy(digits, 0, result, 1, digits.length);
        return result;
    }
    return digits;
}
```

## 67.二进制求和

### 题目

给你两个二进制字符串，返回它们的和（用二进制表示）。输入为 非空 字符串且只包含数字 1 和 0。

### 题解

```java
public String addBinary(String a, String b) {
    int i = a.length() - 1, j = b.length() - 1;

    char[] arrayA = a.toCharArray();
    char[] arrayB = b.toCharArray();
    int carry = 0;

    StringBuilder stringBuilder = new StringBuilder();
    while (i >= 0 || j >= 0) {
        char curA = i >= 0 ? arrayA[i] : '0';
        char curB = j >= 0 ? arrayB[j] : '0';
        if (carry == 1) {
            if (curA != curB) {
                stringBuilder.append('0');
            } else {
                stringBuilder.append('1');

                if (curA == '0') {
                    carry = 0;
                }
            }
        } else {
            if (curA != curB) {
                stringBuilder.append('1');
            } else {
                stringBuilder.append('0');

                if (curA == '1') {
                    carry = 1;
                }
            }
        }
        i--;
        j--;
    }
    if (carry != 0) {
        stringBuilder.append('1');
    }
    return stringBuilder.reverse().toString();
}
```

## 68.文本左右对齐

### 题目

给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。文本的最后一行应为左对齐，且单词之间不插入额外的空格。

### 题解

#### 傻逼题目（不能通过）

```java
private List<String> result = new ArrayList<>();

public List<String> fullJustify(String[] words, int maxWidth) {
    addWord(words, 0, new ArrayList<>(), 0, 0, maxWidth);
    return result;
}

private void addWord(String[] words, int index, List<String> curCol, int curColWordsNumber, int curColLen,
                     int maxWidth) {
    if (index == words.length) {
        if (curColWordsNumber != 0) {
            insertLastColumn(curCol, maxWidth);
        }
        return;
    }

    String cur = words[index];
    int addLen = cur.length() + curColLen;

    if (addLen == maxWidth) {
        curCol.add(cur);
        result.add(toString(curCol, 0, 1, maxWidth));
        addWord(words, index + 1, new ArrayList<>(), 0, 0, maxWidth);
    } else if (addLen < maxWidth) {
        curCol.add(cur);
        addWord(words, index + 1, curCol, curColWordsNumber + 1, curColLen + cur.length() + 1, maxWidth);
    } else {
        // 重排并插入上一行
        // 空格数
        insertColumn(maxWidth, curColLen, curColWordsNumber, curCol);

        List<String> newCurCol = new ArrayList<>();
        newCurCol.add(cur);
        addWord(words, index + 1, newCurCol, 1, cur.length() + 1, maxWidth);
    }
}

private void insertColumn(int maxWidth, int curColLen, int curColWordsNumber, List<String> curCol) {
    // 重排并插入上一行
    if (curColWordsNumber == 1) {
        result.add(toString(curCol, 0, maxWidth - curCol.get(0).length(), maxWidth));
        return;
    }

    // 空格数
    int blankNumbers = maxWidth - curColLen + curColWordsNumber;
    int ave = blankNumbers / (curColWordsNumber - 1);
    int mod = blankNumbers % (curColWordsNumber - 1);
    result.add(toString(curCol, mod, ave, maxWidth));
}

private void insertLastColumn(List<String> curCol, int maxWidth) {
    StringBuilder lastCol = new StringBuilder();
    for (int i = 0; i < curCol.size(); i++) {
        lastCol.append(curCol.get(i));
        lastCol.append(' ');
    }
    int temp = maxWidth - lastCol.length();
    while (temp > 0) {
        lastCol.append(' ');
        temp--;
    }
    result.add(lastCol.toString());
}

private String toString(List<String> list, int mod, int ave, int maxLen) {
    StringBuilder stringBuilder = new StringBuilder();

    for (int i = 0; i < list.size(); i++) {
        if (i < list.size() - 1) {
            stringBuilder.append(list.get(i));
            int temp = ave + (mod > 0 ? 1 : 0);
            mod--;
            while (temp > 0) {
                stringBuilder.append(' ');
                temp--;
            }
        } else {
            stringBuilder.append(list.get(i));
        }
    }
    return stringBuilder.toString();
}
```

#### 评论区解答

```java
public List<String> fullJustify(String[] words, int maxWidth) {
    List<String> list = new ArrayList();
    int n = words.length, i = 0, j = 0, blank = 0, rest, wdCnt;
    while(i < n){
        rest = maxWidth;
        wdCnt = 0;
        blank = 0;
        while(j < n && rest >= words[j].length()){
            rest -= words[j++].length(); 
            wdCnt++;
            rest -= 1;  //如果后面还要接单词，至少要留一个空格
            blank++;
        }
        blank += rest;
        // System.out.println(blank);
        StringBuilder sb = new StringBuilder();
        //特殊情况1， 如果是最后一行， 单词之间只占一个空格
        if(j >= n){
            while(i < j){
                sb.append(words[i++]).append(" ");
            }
            sb.deleteCharAt(sb.length() - 1);
            while(sb.length() < maxWidth){
                sb.append(" ");
            }
        }else if(wdCnt == 1){
            //特殊情况2， 如果一行只有一个单词， 补齐右边的空格
            while(i < j){
                sb.append(words[i++]).append(" ");
            }
            sb.deleteCharAt(sb.length() - 1);
            while(sb.length() < maxWidth){
                sb.append(" ");
            }
        }else{
            //普通情况
            int mod = blank % (wdCnt - 1);
            int bsn = blank / (wdCnt - 1);
            while(i < j){
                sb.append(words[i++]);
                int k = bsn + (mod > 0 ? 1: 0);
                mod--;
                if(i < j) for(int l = 0; l < k; l++) sb.append(" ");
            }
        }
        i = j;
        list.add(sb.toString());
    }
    return list;
}
```

## 69.x的平方根

### 题目

实现sqrt函数，计算并返回x的平方根，结果只保留整数部分

### 题解

#### 渣渣解法

```java
public int mySqrt(int x) {
    int m = 0;

    if (x == 0) {
        return 0;
    } else if (x < 4) {
        return 1;
    }

    int temp = x;
    while (temp > 0) {
        temp = temp >> 2;
        m++;
    }

    int i = (int) Math.pow(2, m - 1);
    while (i <= (int) Math.pow(2, m)) {
        int mul = i * i;
        if (mul > x || mul < 0) {
            return i - 1;
        } else if (mul == x) {
            return i;
        }
        i++;
    }
    return i;
}
```

#### 二分法

```java
public int mySqrt(int x) {
    int l = 0, r = x, ans = -1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if ((long) mid * mid <= x) {
            ans = mid;
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return ans;
}
```

## 70.爬楼梯

### 题目

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

### 题解

```java
public int climbStairs(int n) {
    int[] dp = new int[n];

    for (int i = 0; i < n; i++) {
        if (i == 0 || i == 1) {
            dp[i] = i + 1;
            continue;
        }
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n - 1];
}
```

## 71.简化路径

### 题目

以 Unix 风格给出一个文件的绝对路径，你需要简化它。或者换句话说，将其转换为规范路径。在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。更多信息请参阅：Linux / Unix中的绝对路径 vs 相对路径。请注意，返回的规范路径必须始终以斜杠 / 开头，并且两个目录名之间必须只有一个斜杠 /。最后一个目录名（如果存在）不能以 / 结尾。此外，规范路径必须是表示绝对路径的最短字符串。

### 题解

#### 难点在于读懂题

```java
public String simplifyPath(String path) {
    String[] pathArray = path.split("/");
    StringBuilder stringBuilder = new StringBuilder();

    for (int i = 0; i < pathArray.length; i++) {
        String cur = pathArray[i];

        if (".".equals(cur) || "".equals(cur)) {
            continue;
        } else if ("..".equals(cur)) {
            int j = stringBuilder.length() - 1;
            while (j >= 0) {
                char c = stringBuilder.charAt(j);

                stringBuilder.deleteCharAt(j);
                if (c == '/') {
                    break;
                }
                j--;
            }
        } else {
            stringBuilder.append('/');
            stringBuilder.append(cur);
        }
    }
    return stringBuilder.length() == 0 ? "/" : stringBuilder.toString();
}
```

## 72.编辑距离

### 题目

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。你可以对一个单词进行如下三种操作：

- 插入一个字符；
- 删除一个字符；
- 替换一个字符

### 题解

#### DP（太难了）

```java
public int minDistance(String word1, String word2) {
    if (word1.length() == 0) {
        return word2.length();
    }
    if (word2.length() == 0) {
        return word1.length();
    }

    // dp[i][j] 表示word1的前(i + 1)位与word2的前(j + 1)位之间的编辑距离
    int[][] dp = new int[word1.length()][word2.length()];

    char[] array1 = word1.toCharArray();
    char[] array2 = word2.toCharArray();


    for (int i = 0; i < array1.length; i++) {
        for (int j = 0; j < array2.length; j++) {
            if (i == 0 && j == 0) {
                dp[0][0] = array1[0] == array2[0] ? 0 : 1;
            } else if (j == 0) {
                if (dp[i - 1][0] == i - 1) {
                    dp[i][0] = i;
                } else {
                    dp[i][0] = array1[i] == array2[0] ? i : i + 1;
                }
            } else if (i == 0) {
                if (dp[0][j - 1] == j - 1) {
                    dp[0][j] = j;
                } else {
                    dp[0][j] = array1[0] == array2[j] ? j : j + 1;
                }
            } else {
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1),
                                    array1[i] == array2[j] ? dp[i - 1][j - 1] : (dp[i - 1][j - 1] + 1));
            }
        }
    }
    return dp[array1.length - 1][array2.length - 1];
}
```

## 73.矩阵置零

### 题目

给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法

### 题解

```java
public void setZeroes(int[][] matrix) {
    int m = matrix.length;
    if (m == 0) {
        return;
    }
    int n = matrix[0].length;

    Set<Integer> colSet = new HashSet<>();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == 0) {
                colSet.add(j);
                for (int k = 0; k < i; k++) {
                    matrix[k][j] = 0;
                }

                for (int t = 0; t < n; t++) {
                    if (matrix[i][t] == 0 && t > j) {
                        colSet.add(t);

                        for (int k = 0; k < i; k++) {
                            matrix[k][t] = 0;
                        }
                    }
                    matrix[i][t] = 0;
                }
                j = n;
            } else {
                if (colSet.contains(j)) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
}
```

## 74.搜索二维矩阵

### 题目

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

- 每行中的整数从左到右按升序排列。
- 每行的第一个整数大于前一行的最后一个整数。

### 题解

#### 二分法

```java
public boolean searchMatrix(int[][] matrix, int target) {
    int m = matrix.length;
    if (m == 0) {
        return false;
    }
    int n = matrix[0].length;
    if (n == 0) {
        return false;
    }

    int leftRow = 0, leftCol = 0, rightRow = m - 1, rightCol = n - 1;

    while (leftRow * n + leftCol < rightRow * n + rightCol) {
        int distance = ((leftRow + rightRow) * n + leftCol + rightCol) / 2;
        int midRow = distance / n;
        int midCol = (distance % n);

        if (matrix[midRow][midCol] == target) {
            return true;
        } else if (matrix[midRow][midCol] < target) {
            leftRow = midRow;
            leftCol = midCol;

            if (leftCol == n - 1) {
                leftCol = 0;
                leftRow++;
            } else {
                leftCol++;
            }
        } else {
            rightRow = midRow;
            rightCol = midCol;

            if (rightCol == 0) {
                rightRow--;
                rightCol = n - 1;
            } else {
                rightCol--;
            }
        }
    }
    return matrix[leftRow][leftCol] == target;
}
```

## 75.颜色分类

### 题目

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

### 题解

#### 自己想的

```java
public void sortColors(int[] nums) {
    if (nums.length == 0) {
        return;
    }
    int left = 0, right = nums.length - 1;

    while (left <= right && nums[left] == 0) {
        left++;
    }
    while (right >= left && nums[right] == 2) {
        right--;
    }

    int i = left;
    while (i <= right) {
        if (nums[i] == 0) {
            if (i == left) {
                i++;
            } else {
                nums[i] = nums[left];
                nums[left] = 0;
            }
            left++;
        } else if (nums[i] == 1) {
            i++;
        } else {
            // nums[i] == 2
            nums[i] = nums[right];
            nums[right] = 2;
            right--;
        }
    }
}
```

## 76.最小覆盖子串

### 题目

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 ""

### 题解

#### 滑动窗口

```java
private Map<Character, Integer> tMap = new HashMap<>();

public String minWindow(String s, String t) {
    if (t.length() > s.length()) {
        return "";
    }

    for (char cur : t.toCharArray()) {
        if (tMap.containsKey(cur)) {
            tMap.put(cur, tMap.get(cur) + 1);
        } else {
            tMap.put(cur, 1);
        }
    }

    Map<Character, Integer> windowMap = new HashMap<>();
    windowMap.put(s.charAt(0), 1);

    int left = 0, right = 0, resL = -1, resR = -1;
    while (right < s.length()) {
        if (isMatch(windowMap, tMap)) {
            if (resL == -1 || ((right - left) < (resR - resL))) {
                resL = left;
                resR = right;
            }
            char cur = s.charAt(left);
            windowMap.put(cur, windowMap.get(cur) - 1);
            left++;
        } else {
            right++;

            if (right < s.length()) {
                char cur = s.charAt(right);
                if (windowMap.containsKey(cur)) {
                    windowMap.put(cur, windowMap.get(cur) + 1);
                } else {
                    windowMap.put(cur, 1);
                }
            }
        }
    }
    if (resL == -1) {
        return "";
    } else {
        return s.substring(resL, resR + 1);
    }
}

private boolean isMatch(Map<Character, Integer> windowMap, Map<Character, Integer> tMap) {
    for (Map.Entry<Character, Integer> entry : tMap.entrySet()) {
        char key = entry.getKey();
        if (!windowMap.containsKey(key) || windowMap.get(key) < entry.getValue()) {
            return false;
        }
    }
    return true;
}
```

#### 评论区大佬解法

```java
class Solution {
    public String minWindow(String s, String t) {
        int[] window = new int[128], need = new int[128];
        char[] ss = s.toCharArray(), tt = t.toCharArray();
        int count = 0, min = ss.length;
        String res = "";
        for (int i = 0; i < tt.length; i++) {
            need[tt[i]]++;
        }
        int i = 0, j = 0;
        while(j < ss.length) {
            char c = ss[j];
            window[c]++;
            if (window[c] <= need[c]) count++;
            while(count == tt.length) {
                if (j - i + 1 <= min){
                    res = s.substring(i, j + 1);
                    min = j - i + 1;
                }
                window[ss[i]]--;
                if (window[ss[i]] < need[ss[i]]) count--;
                i++;
                if (i >= ss.length) break;
            }
            j++;
        }
        return res;
    }
}
```

## 77.组合

### 题目

给定两个整数n和k，返回1...n中所有可能的k个数的组合

### 题解

#### easy dfs

```java
private List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> combine(int n, int k) {
    if (k <= n) {
        group(new ArrayList<>(), 1, n, k);
    }
    return result;
}

private void group(List<Integer> tempRes, int num, int n, int k) {
    if (tempRes.size() == k) {
        result.add(new ArrayList<>(tempRes));
        return;
    }
    for (int i = num; i <= n; i++) {
        tempRes.add(i);
        group(tempRes, i + 1, n, k);
        tempRes.remove(tempRes.size() - 1);
    }
}
```

#### 剪枝

```java
private List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> combine(int n, int k) {
    if (k <= n) {
        group(new ArrayList<>(), 1, n, k);
    }
    return result;
}

private void group(List<Integer> tempRes, int num, int n, int k) {
    if (tempRes.size() + n - num + 1 < k) {
        return;
    }

    if (tempRes.size() == k) {
        result.add(new ArrayList<>(tempRes));
        return;
    }
    for (int i = num; i <= n; i++) {
        tempRes.add(i);
        group(tempRes, i + 1, n, k);
        tempRes.remove(tempRes.size() - 1);
    }
}
```

## 78.子集

### 题目

给定一组不含重读元素的整数数组nums，返回该数组所有可能的子集

### 题解

```java
private List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> subsets(int[] nums) {
    group(nums, 0, new ArrayList<>());
    return result;
}

private void group(int[] nums, int index, List<Integer> tempRes) {
    result.add(new ArrayList<>(tempRes));
    if (index == nums.length) {
        return;
    }

    for (int i = index; i < nums.length; i++) {
        tempRes.add(nums[i]);
        group(nums, i + 1, tempRes);
        tempRes.remove(tempRes.size() - 1);
    }
}
```

## 79.单词搜索

### 题目

给定一个二维网格和一个单词，找出该单词是否存在于网格中。单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

### 题解

#### dfs

```java
private boolean exist = false;

public boolean exist(char[][] board, String word) {
    if (board.length == 0 || board[0].length == 0 || word.length() == 0 || board.length * board[0].length < word.length()) {
        return false;
    }
    for (int i = 0; i < board.length; i++) {
        for (int j = 0; j < board[0].length; j++) {
            search(board, word, 0, i, j, null);
        }
    }
    return exist;
}

/**
     * @param wrongDirection 不能走的方向
     */
private void search(char[][] board, String word, int index, int row, int col, String wrongDirection) {
    if (row == -1 || col == -1 || row == board.length || col == board[0].length || board[row][col] == ' ') {
        return;
    }

    if (board[row][col] == word.charAt(index)) {
        if (index == word.length() - 1) {
            exist = true;
        } else {
            char temp = board[row][col];
            board[row][col] = ' ';

            if (wrongDirection == null || !wrongDirection.equals("up")) {
                search(board, word, index + 1, row - 1, col, "down");
                if (exist) return;
            }
            if (wrongDirection == null || !wrongDirection.equals("down")) {
                search(board, word, index + 1, row + 1, col, "up");
                if (exist) return;
            }
            if (wrongDirection == null || !wrongDirection.equals("left")) {
                search(board, word, index + 1, row, col - 1, "right");
                if (exist) return;
            }
            if (wrongDirection == null || !wrongDirection.equals("right")) {
                search(board, word, index + 1, row, col + 1, "left");
                if (exist) return;
            }
            board[row][col] = temp;
        }
    }
}
```

## 80.删除排序数组中的重复项II

### 题目

给定一个增序排列数组 nums ，你需要在 原地 删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

### 题解

```java
public int removeDuplicates(int[] nums) {
    if (nums.length < 3) {
        return nums.length;
    }

    int i = 1, j = 1;
    boolean canSame = true;
    while (j < nums.length) {
        if (canSame) {
            if (nums[j] == nums[j - 1]) {
                canSame = false;
            }
            nums[i++] = nums[j++];
        } else {
            if (nums[j] == nums[j - 1]) {
                // 跳过
                j++;
            } else {
                nums[i++] = nums[j++];
                canSame = true;
            }
        }
    }
    return i;
}
```

## 81.搜索排序数组2

### 题目

假设按照升序排序的数组在预先未知的某个点上进行了旋转。编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。

### 题解

#### 二分

```java
public boolean search(int[] nums, int target) {
    if (nums.length == 0) {
        return false;
    }
    int left = 0, right = nums.length - 1;

    while (left < right) {
        int mid = (left + right) / 2;

        if (nums[left] == nums[mid]) {
            if (nums[left] == target) {
                return true;
            } else {
                left++;
            }
        } else if (nums[left] < nums[mid]) {
            if (nums[mid] == target) {
                return true;
            } else if (target < nums[mid] && target >= nums[left]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (nums[mid] == target) {
                return true;
            } else if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return nums[left] == target;
}
```

## 82.删除排序链表中的重复元素2

### 题目

给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中没有重复出现的数字

### 题解

```java
public ListNode deleteDuplicates(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }

    ListNode rootPre = new ListNode();
    ListNode lastValid = null;
    ListNode checkNode = head;
    int checkNum = checkNode.val;

    ListNode cur = checkNode.next;

    while (cur != null) {
        if (cur.val != checkNum) {
            if (lastValid == null) {
                lastValid = checkNode;
                lastValid.next = null;
                rootPre.next = lastValid;
            } else {
                lastValid.next = checkNode;
                lastValid = checkNode;
                lastValid.next = null;
            }
            checkNode = cur;
            checkNum = checkNode.val;
            cur = cur.next;
        } else {
            while (cur != null && cur.val == checkNum) {
                cur = cur.next;
            }
            if (cur != null) {
                checkNode = cur;
                checkNum = checkNode.val;
                cur = cur.next;
            } else {
                checkNode = null;
            }
        }

        if (cur == null && checkNode != null) {
            if (lastValid == null) {
                return checkNode;
            } else {
                lastValid.next = checkNode;
                return rootPre.next;
            }
        }
    }
    return rootPre.next;
}
```

## 83.删除排序链表中的重复元素

### 题目

给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

### 题解

```java
public ListNode deleteDuplicates(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }

    ListNode rootPre = new ListNode();
    rootPre.next = head;

    ListNode pre = head;
    ListNode cur = head.next;

    while (cur != null) {
        if (cur.val == pre.val) {
            cur = cur.next;
            pre.next = cur;
        } else {
            pre = cur;
            cur = cur.next;
        }
    }
    return rootPre.next;
}
```

## 84.柱状图中最大的矩形

### 题目

给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积

### 题解

#### 暴力法

```java
public int largestRectangleArea(int[] heights) {
    int max = 0;
    for (int i = 0; i < heights.length; i++) {
        // 以每个数为高，向两边延伸
        int cur = heights[i];
        int left = i - 1;
        while (left >= 0 && heights[left] >= cur) {
            left--;
        }
        left = left + 1;

        int right = i + 1;
        while (right < heights.length && heights[right] >= cur) {
            right++;
        }
        right = right - 1;

        max = Math.max(max, (right - left + 1) * cur);
    }
    return max;
}
```

#### 使用单调栈

```java
public int largestRectangleArea(int[] heights) {
    int len = heights.length;
    int[] left = new int[len];
    int[] right = new int[len];

    Deque<Integer> stack = new ArrayDeque<>();
    for (int i = 0; i < len; i++) {
        int cur = heights[i];
        while (!stack.isEmpty() && heights[stack.peek()] >= cur) {
            stack.pop();
        }
        left[i] = stack.isEmpty() ? -1 : stack.peek();
        stack.push(i);
    }

    stack.clear();
    for (int i = len - 1; i >= 0; i--) {
        int cur = heights[i];
        while (!stack.isEmpty() && heights[stack.peek()] >= cur) {
            stack.pop();
        }
        right[i] = stack.isEmpty() ? len : stack.peek();
        stack.push(i);
    }
    int res = 0;
    for (int i = 0; i < len; i++) {
        res = Math.max(res, heights[i] * (right[i] - 1 - (left[i] + 1) + 1));
    }
    return res;
}
```

## 85.最大矩形

### 题目

给定一个仅包含0和1、大小为rows x cols的二维二进制矩阵，找出只包含1的最大矩阵，并返回其面积

### 题解

#### 记录每个点的最大宽度，然后横着看同84

```java
public int maximalRectangle(char[][] matrix) {
    int rowLen = matrix.length;

    if (rowLen == 0) {
        return 0;
    }
    int colLen = matrix[0].length;

    // 记录每个点的最大宽度，然后从又往左就是和柱状图求最大面积一样了
    int[][] maxWidth = new int[rowLen][colLen];
    for (int i = 0; i < rowLen; i++) {
        int count = 0;
        for (int j = 0; j < colLen; j++) {
            char cur = matrix[i][j];
            if (cur == '1') {
                count++;
            } else {
                count = 0;
            }
            maxWidth[i][j] = count;
        }
    }

    // 同柱状图求最大面积
    int max = 0;
    for (int col = 0; col < colLen; col++) {
        int[] low = new int[rowLen];
        int[] height = new int[rowLen];

        Deque<Integer> stack = new ArrayDeque<>();
        for (int row = rowLen - 1; row >= 0; row--) {
            int cur = maxWidth[row][col];
            while (!stack.isEmpty() && maxWidth[stack.peek()][col] >= cur) {
                stack.pop();
            }
            low[row] = stack.isEmpty() ? rowLen : stack.peek();
            stack.push(row);
        }
        stack.clear();
        for (int row = 0; row < rowLen; row++) {
            int cur = maxWidth[row][col];
            while (!stack.isEmpty() && maxWidth[stack.peek()][col] >= cur) {
                stack.pop();
            }
            height[row] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(row);
        }

        for (int row = rowLen - 1; row >= 0; row--) {
            max = Math.max(max, maxWidth[row][col] * (low[row] - height[row] - 1));
        }
    }
    return max;
}
```

## 86.分隔链表

### 题目

给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。你应当保留两个分区中每个节点的初始相对位置。

### 题解

#### 加指针就完事了

```java
public ListNode partition(ListNode head, int x) {
    if (head == null || head.next == null) {
        return head;
    }
    ListNode preRoot = new ListNode();
    preRoot.next = head;
    ListNode pre = preRoot;

    ListNode midNodePre = null;

    while (head != null) {
        if (midNodePre != null) {
            if (head.val < x) {
                ListNode temp = midNodePre.next;
                ListNode tempHead = head;
                head = head.next;
                pre.next = head;
                midNodePre.next = tempHead;
                tempHead.next = temp;
                midNodePre = tempHead;
            } else {
                pre = head;
                head = head.next;
            }
        } else {
            if (head.val < x) {
                pre = head;
                head = head.next;
            } else {
                midNodePre = pre;
                pre = head;
                head = head.next;
            }
        }
    }
    return preRoot.next;
}
```

## 87.扰乱字符串

### 题目

给定一个字符串 s1，我们可以把它递归地分割成两个非空子字符串，从而将其表示为二叉树。在扰乱这个字符串的过程中，我们可以挑选任何一个非叶节点，然后交换它的两个子节点。给出两个长度相等的字符串 *s1* 和 *s2*，判断 *s2* 是否是 *s1* 的扰乱字符串。

### 题解

#### dp

```java
public boolean isScramble(String s1, String s2) {
    if (s1.length() == 0 || s1.length() != s2.length()) {
        return false;
    }
    int len = s1.length();
    boolean[][][] dp = new boolean[len][len][len + 1];

    for (int l = 1; l <= len; l++) {
        for (int i = 0; i <= len - l; i++) {
            for (int j = 0; j <= len - l; j++) {
                if (l == 1) {
                    dp[i][j][1] = s1.charAt(i) == s2.charAt(j);
                } else {
                    for (int mid = 1; mid <= l / 2; mid++) {
                        if ((dp[i][j][mid] && dp[i + mid][j + mid][l - mid]) ||
                            (dp[i][j + l - mid][mid] && dp[i + mid][j][l - mid])) {
                            dp[i][j][l] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    return dp[0][0][len];
}
```

## 88.合并两个有序数组

### 题目

给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

### 题解

#### 从后往前（从前往后想了好久）

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int i = m - 1, j = n - 1;
    int p = m + n - 1;

    while (i >= 0 && j >= 0) {
        nums1[p--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--];
    }
    if (j >= 0) {
        System.arraycopy(nums2, 0, nums1, 0, j + 1);
    }
}
```

## 89.格雷编码

### 题目

格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。即使有多个不同答案，你也只需要返回其中一种。格雷编码序列必须以 0 开头。

### 题解

#### 把n-1的结果倒过来

```java
public List<Integer> grayCode(int n) {
    List<Integer> result = new ArrayList<>();
    if (n == 0) {
        result.add(0);
        return result;
    } else if (n == 1) {
        result.add(0);
        result.add(1);
        return result;
    } else {
        List<Integer> lastResult = grayCode(n - 1);
        for (int i = lastResult.size() - 1; i >= 0; i--) {
            int temp = lastResult.get(i);
            lastResult.add((int) Math.pow(2, n - 1) + temp);
        }
        return lastResult;
    }
}
```

## 90.子集2

### 题目

给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。说明：解集不能包含重复的子集。

### 题解

#### 老朋友（注意remove）

```java
private List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> subsetsWithDup(int[] nums) {
    List<Integer> tempResult = new ArrayList<>();
    Arrays.sort(nums);
    group(tempResult, 0, nums);
    return result;
}

private void group(List<Integer> tempResult, int index, int[] nums) {
    result.add(new ArrayList<>(tempResult));
    if (index == nums.length) {
        return;
    }
    for (int i = index; i < nums.length; i++) {
        if (i == index || nums[i] != nums[i - 1]) {
            tempResult.add(nums[i]);
            group(tempResult, i + 1, nums);
            tempResult.remove(tempResult.size() - 1);
        }
    }
}
```

## 91.解码方法

### 题解

一条包含字母 A-Z 的消息通过以下方式进行了编码：编为1-26。给定一个只包含数字的非空字符串，请计算解码方法的总数。题目数据保证答案肯定是一个 32 位的整数。

### 题解

#### 看似简单，实则很难（抄的答案）

dfs会超时

```java
public int numDecodings(String s) {
    if (s.charAt(0) == '0') {
        return 0;
    }
    int pre = 1, curr = 1;
    for (int i = 1; i < s.length(); i++) {
        int temp = curr;
        if (s.charAt(i) == '0') {
            if (s.charAt(i - 1) == '1' || s.charAt(i - 1) == '2') {
                temp = pre;
            } else {
                return 0;
            }
        } else if ((s.charAt(i - 1) == '1' && s.charAt(i) != '0') ||
                   (s.charAt(i - 1) == '2' && s.charAt(i) > '0' && s.charAt(i) <= '6')) {
            temp += pre;
        }
        pre = curr;
        curr = temp;
    }
    return curr;
}
```

## 92.反转链表2

### 题目

反转从位置 *m* 到 *n* 的链表。请使用一趟扫描完成反转。

### 题解

```java
public ListNode reverseBetween(ListNode head, int m, int n) {
    ListNode preRoot = new ListNode();
    preRoot.next = head;

    ListNode nodeMPre = preRoot;
    ListNode pre = preRoot;
    int index = 1;
    while (head != null) {
        if (index == m) {
            nodeMPre = pre;
            pre = head;
            head = head.next;
        } else if (index > m && index <= n) {
            ListNode temp = head.next;
            head.next = nodeMPre.next;
            nodeMPre.next = head;
            pre.next = temp;
            head = temp;
        } else {
            pre = head;
            head = head.next;
        }
        index++;
    }
    return preRoot.next;
}
```

## 93.复原IP地址

### 题目

给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。有效的 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。例如："0.1.2.201" 和 "192.168.1.1" 是 有效的 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效的 IP 地址。

### 题解

#### backtrace

```java
public List<String> restoreIpAddresses(String s) {
    List<String> result = new ArrayList<>();

    if (s.length() >= 4) {
        char[] array = s.toCharArray();
        backTrace(array, 0, new StringBuilder(), 0, result);
    }
    return result;
}

private void backTrace(char[] array, int count, StringBuilder stringBuilder, int index, List<String> result) {
    if (count == 3) {
        if (index == array.length || array.length - index > 3) {
            return;
        }
        if (array.length - index > 1 && array[index] == '0') {
            return;
        }
        if (array.length - index == 3 && (array[index] > '2' || (array[index] == '2' && array[index + 1] > '5') ||
                                          (array[index] == '2' && array[index + 1] == '5' && array[index + 2] > '5'))) {
            return;
        }

        stringBuilder.append(String.valueOf(array, index, array.length - index));
        int length = stringBuilder.length();
        result.add(stringBuilder.toString());
        stringBuilder.delete(length - (array.length - index), length);
        return;
    }
    if (array[index] == '0') {
        stringBuilder.append('0');
        stringBuilder.append('.');
        int length = stringBuilder.length();

        backTrace(array, count + 1, stringBuilder, index + 1, result);
        stringBuilder.delete(length - 2, length);
    } else {
        int lenMax = Math.min(3, array.length - (3 - count) - index);
        for (int len = 1; len <= lenMax; len++) {
            if (len == 3) {
                if (array[index] > '2' || (array[index] == '2' && array[index + 1] > '5') ||
                    (array[index] == '2' && array[index + 1] == '5' && array[index + 2] > '5')) {
                    return;
                }
            }

            stringBuilder.append(String.valueOf(array, index, len));
            stringBuilder.append('.');
            int length = stringBuilder.length();
            backTrace(array, count + 1, stringBuilder, index + len, result);
            stringBuilder.delete(length - len - 1, length);
        }
    }
}
```

## 94.二叉树的中序遍历

### 题目

给定一个二叉树的根节点 root ，返回它的 中序 遍历。

### 题解

#### 递归

```java
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> result = new ArrayList<>();

    if (root != null) {
        if (root.left != null) {
            result.addAll(inorderTraversal(root.left));
        }
        result.add(root.val);
        if (root.right != null) {
            result.addAll(inorderTraversal(root.right));
        }
    }
    return result;
}
```

#### 迭代

```java
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> result = new ArrayList<>();

    Deque<TreeNode> stack = new ArrayDeque<>();
    while (root != null || !stack.isEmpty()) {
        while (root != null) {
            stack.push(root);
            root = root.left;
        }
        root = stack.pop();
        result.add(root.val);
        root = root.right;
    }

    return result;
}
```

## 95.不同的二叉搜索树2

### 题目

给定一个整数 *n*，生成所有由 1 ... *n* 为节点所组成的 **二叉搜索树** 

### 题解

#### 递归

```java
public List<TreeNode> generateTrees(int n) {
    return generateTrees(1, n);
}

private List<TreeNode> generateTrees(int m, int n) {
    List<TreeNode> result = new ArrayList<>();
    if (m > n) {
        return result;
    } else if (m == n) {
        result.add(new TreeNode(m));
        return result;
    } else {
        for (int i = m; i <= n; i++) {
            List<TreeNode> treeLeft = generateTrees(m, i - 1);
            List<TreeNode> treeRight = generateTrees(i + 1, n);

            if (treeLeft.isEmpty()) {
                for (TreeNode nodeRight : treeRight) {
                    result.add(new TreeNode(i, null, nodeRight));
                }
            } else if (treeRight.isEmpty()) {
                for (TreeNode nodeLeft : treeLeft) {
                    result.add(new TreeNode(i, nodeLeft, null));
                }
            } else {
                for (TreeNode nodeLeft : treeLeft) {
                    for (TreeNode nodeRight : treeRight) {
                        result.add(new TreeNode(i, nodeLeft, nodeRight));
                    }
                }
            }
        }
    }
    return result;
}
```

## 96.不同的二叉搜索树

### 题目

给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种

### 题解

#### dp

```java
public int numTrees(int n) {
    // 1,2与3,4能构成的二叉搜索树的数量是一样的
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            dp[i] += dp[j - 1] * dp[i - j];
        }
    }
    return dp[n];
}
```

## 97.交错字符串

### 题目

给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
提示：a + b 意味着字符串 a 和 b 连接。

### 题解

#### dp

```java
public boolean isInterleave(String s1, String s2, String s3) {
    if (s1.length() + s2.length() != s3.length()) {
        return false;
    }

    boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
    for (int i = 0; i <= s1.length(); i++) {
        for (int j = 0; j <= s2.length(); j++) {
            if (i == 0 && j == 0) {
                dp[0][0] = true;
            } else if (i == 0) {
                dp[0][j] = s2.substring(0, j).equals(s3.substring(0, j));
            } else if (j == 0) {
                dp[i][0] = s1.substring(0, i).equals(s3.substring(0, i));
            } else {
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1))
                    || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
            }
        }
    }
    return dp[s1.length()][s2.length()];
}
```

## 98.验证二叉搜索树

### 题目

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

### 题解

#### 递归

```java
public boolean isValidBST(TreeNode root) {
    if (root == null) {
        return true;
    }
    TreeNode leftNode = root.left;
    if (leftNode != null) {
        while (leftNode.right != null) {
            leftNode = leftNode.right;
        }
    }

    TreeNode rightNode = root.right;
    if (rightNode != null) {
        while (rightNode.left != null) {
            rightNode = rightNode.left;
        }
    }

    return (leftNode == null || (root.val > leftNode.val && isValidBST(root.left))) &&
        (rightNode == null || (root.val < rightNode.val && isValidBST(root.right)));
}
```

## 99.恢复二叉搜索树

### 题目

给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

### 题解

```java
public void recoverTree(TreeNode root) {
    List<Integer> nums = new ArrayList<>();
    throughTreeNode(root, nums);

    int leftValue = -1, rightValue = -1;

    for (int i = 0; i < nums.size() - 1; i++) {
        if (nums.get(i) > nums.get(i + 1)) {
            rightValue = nums.get(i + 1);
            if (leftValue == -1) {
                leftValue = nums.get(i);
            } else {
                break;
            }
        }
    }
    swapNode(root, leftValue, rightValue);
}

private void swapNode(TreeNode root, int leftValue, int rightValue) {
    if (root == null) {
        return;
    }

    swapNode(root.left, leftValue, rightValue);
    if (root.val == leftValue) {
        root.val = rightValue;
    } else if (root.val == rightValue) {
        root.val = leftValue;
        return;
    }
    swapNode(root.right, leftValue, rightValue);
}

private void throughTreeNode(TreeNode root, List<Integer> nums) {
    if (root == null) {
        return;
    }
    throughTreeNode(root.left, nums);
    nums.add(root.val);
    throughTreeNode(root.right, nums);
}
```

## 100.相同的树

### 题目

给定两个二叉树，编写一个函数来检验它们是否相同。如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

### 题解

#### 递归

```java
public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p == null) {
        return q == null;
    } else if (q == null) {
        return false;
    } else {
        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```