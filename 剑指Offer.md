# 剑指Offer

## 1.数组中重复的数字（3）

### 题目

找出数组中重复的数字。在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

### 题解

```java
public int findRepeatNumber(int[] nums) {
    boolean hasZero = false;
    for (int i = 0; i < nums.length; i++) {
        int temp = nums[i];
        if (temp < 0) {
            temp = -temp;
        }
        if (nums[temp] < 0) {
            return temp;
        } else if (nums[temp] == 0) {
            if (hasZero) {
                return temp;
            } else {
                hasZero = true;
            }
        } else {
            nums[temp] = -nums[temp];
        }
    }
    throw new IllegalArgumentException();
}
```

## 2.二维数组中的查找（4）

### 题目

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

### 题解

```java
public boolean findNumberIn2DArray(int[][] matrix, int target) {
    int rowLen = matrix.length;
    if (rowLen == 0) {
        return false;
    }
    int colLen = matrix[0].length;

    int i = rowLen - 1, j = 0;
    while (i >= 0 && j < colLen) {
        if (matrix[i][j] == target) {
            return true;
        } else if (matrix[i][j] > target) {
            i--;
        } else {
            j++;
        }
    }
    return false;
}
```

## 3.替换空格（5）

### 题目

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

### 题解

```java
public String replaceSpace(String s) {
    String str = "%20";
    StringBuilder stringBuilder = new StringBuilder();

    for (int i = 0; i < s.length(); i++) {
        char temp = s.charAt(i);

        if (temp == ' ') {
            stringBuilder.append(str);
        } else {
            stringBuilder.append(temp);
        }
    }
    return stringBuilder.toString();
}
```

## 4.从尾到头打印链表（6）

### 题目

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

### 题解

```java
public int[] reversePrint(ListNode head) {
    Deque<Integer> stack = new ArrayDeque<>();

    int count = 0;
    while (head != null) {
        stack.push(head.val);
        count++;
        head = head.next;
    }
    int[] result = new int[count];
    int i = 0;
    while (!stack.isEmpty()) {
        result[i] = stack.pop();
        i++;
    }
    return result;
}
```

## 5.重建二叉树（7）

### 题目

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

### 题解

#### 递归

```java
public TreeNode buildTree(int[] preorder, int[] inorder) {
    return buildTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
}

private TreeNode buildTree(int[] preorder, int preLeft, int preRight, int[] inorder, int inLeft, int inRight) {
    System.out.println("preLeft: " + preLeft + ", preRight: " + preRight + ", inLeft: " + inLeft + ", inRight: " + inRight);

    if (preRight < preLeft || inRight < inLeft) {
        return null;
    }

    TreeNode root = new TreeNode(preorder[preLeft]);

    for (int i = inLeft; i <= inRight; i++) {
        if (root.val == inorder[i]) {
            root.left = buildTree(preorder, preLeft + 1, preLeft + i - inLeft, inorder, inLeft, i - 1);
            root.right = buildTree(preorder, preLeft - inLeft + i + 1, preRight, inorder, i + 1, inRight);
            return root;
        }
    }
    throw new IllegalArgumentException();
}
```

## 6.用两个栈实现队列（9）

### 题目

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

### 题解

```java
private Deque<Integer> stack1 = new ArrayDeque<>();
private Deque<Integer> stack2 = new ArrayDeque<>();

public CQueue() {

}

public void appendTail(int value) {
    stack1.push(value);
}

public int deleteHead() {
    if (!stack2.isEmpty()) {
        return stack2.pop();
    } else {
        while (!stack1.isEmpty()) {
            stack2.push(stack1.pop());
        }
        return stack2.isEmpty() ? -1 : stack2.pop();
    }
}
```

## 7.斐波拉契数列（10-1）

### 题目

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项

### 题解

```java
public int fib(int n) {
    int mod = 1000000007;

    if (n == 0 || n == 1) {
        return n;
    }

    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 1;

    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
        if (dp[i] >= mod) {
            dp[i] = dp[i] % mod;
        }
    }
    return dp[n];
}
```

## 8.青蛙跳台阶问题（10-2）

### 题目

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

### 题解

```java
public int numWays(int n) {
    int mod = 1000000007;

    if (n == 0) {
        return 1;
    }

    if (n == 1 || n == 2) {
        return n;
    }

    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;

    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
        if (dp[i] >= mod) {
            dp[i] = dp[i] % mod;
        }
    }
    return dp[n];
}
```

## 9.旋转数组的最小数字（11）

### 题目

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

### 题解

```java
public int minArray(int[] numbers) {
    int left = 0, right = numbers.length - 1;

    int temp = numbers[0];
    while (left < right) {
        int mid = (left + right) / 2;
        if (numbers[mid] == numbers[left] || numbers[left] == numbers[right]) {
            if (mid == left) {
                temp = Math.min(temp, numbers[left]);
            }
            left = left + 1;
        } else if (numbers[mid] > numbers[left] && numbers[left] > numbers[right]) {
            left = mid + 1;
        } else {
            temp = Math.min(temp, numbers[mid]);
            right = mid - 1;
        }
    }
    return Math.min(numbers[left], temp);
}
```

## 10.矩阵中的路径（12）

### 题目

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出

### 题解

```java
public boolean exist(char[][] board, String word) {
    char[] wordArray = word.toCharArray();
    for (int i = 0; i < board.length; i++) {
        for (int j = 0; j < board[0].length; j++) {
            if (exist(board, wordArray, 0, i, j)) {
                return true;
            }
        }
    }
    return false;
}

private boolean exist(char[][] board, char[] wordArray, int index, int i, int j) {
    if (i < 0 || i == board.length || j < 0 || j == board[0].length || board[i][j] != wordArray[index]) {
        return false;
    }
    if (index == wordArray.length - 1) {
        return true;
    }

    board[i][j] = ' ';
    boolean res = exist(board, wordArray, index + 1, i + 1, j) || exist(board, wordArray, index + 1, i,
                                                                        j + 1) || exist(board, wordArray, index + 1, i - 1, j) || exist(board, wordArray, index + 1,
                                                                                                                                        i, j - 1);
    board[i][j] = wordArray[index];
    return res;
}
```

## 11.机器人的运动范围（13）

### 题目

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

### 题解

#### dfs

```java
private int count = 0;

private boolean[][] board;

public int movingCount(int m, int n, int k) {
    board = new boolean[m][n];
    dfs(0, 0, k, m, n);
    return count;
}

private void dfs(int i, int j, int k, int m, int n) {
    if (i == m || j == n || (i / 10 + i % 10 + j / 10 + j % 10) > k || board[i][j]) {
        return;
    }
    count++;
    board[i][j] = true;
    dfs(i + 1, j, k, m, n);
    dfs(i, j + 1, k, m, n);
}
```

## 12.剪绳子（14-1）

### 题目

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18，2 <=n <= 58。

### 题解

#### dp

```java
public int cuttingRope(int n) {
    // dp[n]表示长度为n的绳子，至少切一次的结果
    int[] dp = new int[n + 1];

    dp[1] = 1;
    dp[2] = 1;
    for (int i = 3; i <= n; i++) {
        for (int k = 2; k < i; k++) {
            dp[i] = Math.max(dp[i], Math.max(dp[i - k] * k, (i - k) * k));
        }
    }
    return dp[n];
}
```

#### 贪心

```java
// 切3最优
public int cuttingRope(int n) {
    if (n == 1 || n == 2) {
        return 1;
    }
    if (n == 3) {
        return 2;
    }
    if (n == 4) {
        return 4;
    }

    int res = 1;
    while (n > 4) {
        n = n - 3;
        res = res * 3;
    }
    return res * n;
}
```

## 13.剪绳子2（14-2）

### 题目

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18，2 <=n <= 1000。答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

### 题解

#### 贪心

```java
public int cuttingRope(int n) {
    int mod = 1000000007;

    if (n == 1 || n == 2) {
        return 1;
    }
    if (n == 3) {
        return 2;
    }
    if (n == 4) {
        return 4;
    }

    long res = 1;
    while (n > 4) {
        n = n - 3;
        res = (res * 3) % mod;
    }
    return (int) (res * n % mod);
}
```

## 14.合并两个排序链表

### 题目

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

### 题解

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode preRoot = new ListNode();

    ListNode pre = preRoot;
    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            pre.next = l1;
            pre = pre.next;
            l1 = l1.next;
        } else {
            pre.next = l2;
            pre = pre.next;
            l2 = l2.next;
        }
    }
    if (l1 != null) {
        pre.next = l1;
    } else {
        pre.next = l2;
    }
    return preRoot.next;
}
```

## 15.树的子结构

### 题目

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)。B是A的子结构， 即 A中有出现和B相同的结构和节点值。

### 题解

#### 递归

```java
public boolean isSubStructure(TreeNode A, TreeNode B) {
    if (A == null || B == null) {
        return false;
    }
    return isSub(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
}

/**
     * B为A的以当前节点开始的子树
     */
private boolean isSub(TreeNode A, TreeNode B) {
    if (B == null) {
        return true;
    }
    if (A == null || A.val != B.val) {
        return false;
    }
    return isSub(A.left, B.left) && isSub(A.right, B.right);
}
```

## 16.二叉树的镜像

### 题目

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

### 题解

```java
public TreeNode mirrorTree(TreeNode root) {
    if (root != null) {
        TreeNode temp = root.right;
        root.right = mirrorTree(root.left);
        root.left = mirrorTree(temp);
    }
    return root;
}
```

## 17.对称的二叉树

### 题目

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

### 题解

```java
public boolean isSymmetric(TreeNode root) {
    return root == null || isMirror(root.left, root.right);
}

private boolean isMirror(TreeNode node1, TreeNode node2) {
    if (node1 != null && node2 != null) {
        return node1.val == node2.val && isMirror(node1.left, node2.right)
            && isMirror(node1.right, node2.left);
    }
    return node1 == node2;
}
```

## 18.表示数值的字符串

### 题目

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是

### 题解

shabi题目，".3"和"3."也算合法，还允许有空格，题目没说清楚

#### 状态机

| 状态\下一个字符           | 0-9  | +/-  | e/E  | .    |
| ------------------------- | ---- | ---- | ---- | ---- |
| 0（初始）                 | 1    | 2    | -1   | 4    |
| 1（合法）                 | 1    | -1   | 3    | 4    |
| 2（前一位是正负号）       | 1    | -1   | -1   | -1   |
| 3（前一位是e）            | 5    | 5    | -1   | -1   |
| 4（前一位是小数点）       | 6    | -1   | -1   | -1   |
| 5（合法且剩下必须全数字） | 5    | -1   | -1   | -1   |
| 6（合法且已有小数点）     | 6    | -1   | 3    | -1   |

#### 个人写的没过的解

因为"3."没过，恶心人

```java
public boolean isNumber(String s) {
    if (s == null || s.length() == 0) {
        return false;
    }

    int left = 0;
    while (left < s.length() && s.charAt(left) == ' ') {
        left++;
    }
    if (left == s.length()) {
        return false;
    }
    int right = s.length() - 1;
    while (right >= 0 && s.charAt(right) == ' ') {
        right--;
    }
    if (left == right) {
        return s.charAt(left) >= '0' && s.charAt(left) <= '9';
    }

    int curState;
    char c0 = s.charAt(left);
    if (c0 >= '0' && c0 <= '9') {
        curState = 1;
    } else if (c0 == '+' || c0 == '-') {
        curState = 2;
    } else if (c0 == '.') {
        curState = 4;
    } else {
        return false;
    }

    int[] stateArray1 = {1, -1, 3, 4};
    int[] stateArray2 = {1, -1, -1, -1};
    int[] stateArray3 = {5, 5, -1, -1};
    int[] stateArray4 = {6, -1, -1, -1};
    int[] stateArray5 = {5, -1, -1, -1};
    int[] stateArray6 = {6, -1, 3, -1};

    int[][] stateTransfer = {stateArray1, stateArray2, stateArray3, stateArray4, stateArray5,
                             stateArray6};


    for (int i = 1; i <= right; i++) {
        char c = s.charAt(i);
        int[] stateArray = stateTransfer[curState - 1];

        int index = -1;
        if (c >= '0' && c <= '9') {
            index = 0;
        } else if (c == '+' || c == '-') {
            index = 1;
        } else if (c == 'E' || c == 'e') {
            index = 2;
        } else if (c == '.') {
            index = 3;
        }
        curState = stateArray[index];
        if (curState == -1) {
            return false;
        }
    }
    return curState == 1 || curState == 5 || curState == 6;
}
```

#### 抄的答案

```java
public boolean isNumber(String s) {
        Map<State, Map<CharType, State>> transfer = new HashMap<State, Map<CharType, State>>();
        Map<CharType, State> initialMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_SPACE, State.STATE_INITIAL);
            put(CharType.CHAR_NUMBER, State.STATE_INTEGER);
            put(CharType.CHAR_POINT, State.STATE_POINT_WITHOUT_INT);
            put(CharType.CHAR_SIGN, State.STATE_INT_SIGN);
        }};
        transfer.put(State.STATE_INITIAL, initialMap);
        Map<CharType, State> intSignMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_INTEGER);
            put(CharType.CHAR_POINT, State.STATE_POINT_WITHOUT_INT);
        }};
        transfer.put(State.STATE_INT_SIGN, intSignMap);
        Map<CharType, State> integerMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_INTEGER);
            put(CharType.CHAR_EXP, State.STATE_EXP);
            put(CharType.CHAR_POINT, State.STATE_POINT);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_INTEGER, integerMap);
        Map<CharType, State> pointMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_FRACTION);
            put(CharType.CHAR_EXP, State.STATE_EXP);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_POINT, pointMap);
        Map<CharType, State> pointWithoutIntMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_FRACTION);
        }};
        transfer.put(State.STATE_POINT_WITHOUT_INT, pointWithoutIntMap);
        Map<CharType, State> fractionMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_FRACTION);
            put(CharType.CHAR_EXP, State.STATE_EXP);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_FRACTION, fractionMap);
        Map<CharType, State> expMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_EXP_NUMBER);
            put(CharType.CHAR_SIGN, State.STATE_EXP_SIGN);
        }};
        transfer.put(State.STATE_EXP, expMap);
        Map<CharType, State> expSignMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_EXP_NUMBER);
        }};
        transfer.put(State.STATE_EXP_SIGN, expSignMap);
        Map<CharType, State> expNumberMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_EXP_NUMBER);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_EXP_NUMBER, expNumberMap);
        Map<CharType, State> endMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_END, endMap);

        int length = s.length();
        State state = State.STATE_INITIAL;

        for (int i = 0; i < length; i++) {
            CharType type = toCharType(s.charAt(i));
            if (!transfer.get(state).containsKey(type)) {
                return false;
            } else {
                state = transfer.get(state).get(type);
            }
        }
        return state == State.STATE_INTEGER || state == State.STATE_POINT || state == State.STATE_FRACTION || state == State.STATE_EXP_NUMBER || state == State.STATE_END;
    }

    public CharType toCharType(char ch) {
        if (ch >= '0' && ch <= '9') {
            return CharType.CHAR_NUMBER;
        } else if (ch == 'e' || ch == 'E') {
            return CharType.CHAR_EXP;
        } else if (ch == '.') {
            return CharType.CHAR_POINT;
        } else if (ch == '+' || ch == '-') {
            return CharType.CHAR_SIGN;
        } else if (ch == ' ') {
            return CharType.CHAR_SPACE;
        } else {
            return CharType.CHAR_ILLEGAL;
        }
    }

    enum State {
        STATE_INITIAL,
        STATE_INT_SIGN,
        STATE_INTEGER,
        STATE_POINT,
        STATE_POINT_WITHOUT_INT,
        STATE_FRACTION,
        STATE_EXP,
        STATE_EXP_SIGN,
        STATE_EXP_NUMBER,
        STATE_END,
    }

    enum CharType {
        CHAR_NUMBER,
        CHAR_EXP,
        CHAR_POINT,
        CHAR_SIGN,
        CHAR_SPACE,
        CHAR_ILLEGAL,
    }
```

## 19.调整数组顺序使奇数位于偶数前面

### 题目

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分

### 题解

```java
public int[] exchange(int[] nums) {
    int i = 0, j = nums.length - 1;
    while (i <= j) {
        if (nums[i] % 2 == 1) {
            i++;
            continue;
        }
        if (nums[j] % 2 == 0) {
            j--;
            continue;
        }
        if (nums[i] % 2 == 0) {
            int temp = nums[i];
            nums[i++] = nums[j];
            nums[j--] = temp;
            continue;
        }

        if (nums[j] % 2 == 1) {
            int temp = nums[j];
            nums[j--] = nums[i];
            nums[i++] = temp;
        }
    }
    return nums;
}
```

## 20.二进制中1的个数

### 题目

请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

### 题解

#### 常规解法（会超时，这也能超时，服了）

```java
public int hammingWeight(int n) {
    int count = 0;
    while(n != 0) {
        count += n % 2;
        n = n >> 1;
    }
    return count;
}
```

#### n&(n-1)

**消除右边的1**

```java
public int hammingWeight(int n) {
    int res = 0;
    while(n != 0) {
        res++;
        n &= n - 1;
    }
    return res;
}
```

## 21.顺时针打印矩阵

### 题目

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

### 题解

```java
public int[] spiralOrder(int[][] matrix) {
    int rowLen = matrix.length;
    if (rowLen == 0) {
        return new int[0];
    }

    int colLen = matrix[0].length;
    int total = rowLen * colLen;
    int[] res = new int[total];

    int i = 0, j = 0, cycle = 1, index = 0;
    while (index < total) {
        if (2 * cycle - rowLen == 1) {
            while (j <= colLen - cycle) {
                res[index++] = matrix[i][j];
                j++;
            }
            break;
        }
        if (2 * cycle - colLen == 1) {
            while (i <= rowLen - cycle) {
                res[index++] = matrix[i][j];
                i++;
            }
            break;
        }

        while (j < colLen - cycle) {
            res[index++] = matrix[i][j];
            j++;
        }
        while (i < rowLen - cycle) {
            res[index++] = matrix[i][j];
            i++;
        }
        while (j > cycle - 1) {
            res[index++] = matrix[i][j];
            j--;
        }
        while (i > cycle - 1) {
            res[index++] = matrix[i][j];
            i--;
        }
        i++;
        j++;
        cycle++;
    }
    return res;
}
```

## 22.链表中的倒数第k个节点

### 题目

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

### 题解

```java
public ListNode getKthFromEnd(ListNode head, int k) {
    ListNode preRoot = new ListNode();
    preRoot.next = head;

    ListNode node1 = preRoot;
    ListNode node2 = preRoot;

    while(k > 1 && node1 != null) {
        node1 = node1.next;
        k--;
    }
    while(true) {
        node1 = node1.next;

        if (node1 == null) {
            break;
        }
        node2 = node2.next;
    }
    return node2;
}
```

## 23.数值的整数次方

### 题目

实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题

### 题解

```java
public double myPow(double x, int n) {
    if (n == 0) {
        return 1.0;
    } else if (n < 0) {
        if (n == Integer.MIN_VALUE) {
            return myPow(x, n + 1) / x;
        }
        return 1 / myPow(x, -n);
    } else {
        double temp = myPow(x, n / 2);
        if (n % 2 == 0) {
            return temp * temp;
        } else {
            return temp * temp * x;
        }
    }
}
```

## 24.打印从1到最大的n位数

### 题目

输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

### 题解

```java
public int[] printNumbers(int n) {
    if (n > 10) {
        throw new IllegalArgumentException();
    }
    int maxValue;
    if (n == 10) {
        maxValue = Integer.MAX_VALUE;
    } else {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            builder.append('9');
        }
        maxValue = Integer.valueOf(builder.toString());
    }
    int[] nums = new int[maxValue];
    for (int i = 1; i <= maxValue; i++) {
        nums[i - 1] = i;
    }
    return nums;
}
```

## 25.反转链表（有本事面试考这个）

### 题目

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

### 题解

#### 递归

```java
public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }

    ListNode res = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return res;
}
```

#### 迭代

```java
public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }

    ListNode preRoot = new ListNode();
    preRoot.next = head;

    ListNode pre = preRoot;
    ListNode cur = head.next;
    head.next = null;

    while(cur != null) {
        ListNode temp = cur.next;
        cur.next = pre.next;
        pre.next = cur;
        cur = temp;
    }
    return preRoot.next;
}
```

## 26.删除链表的节点

### 题目

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回删除后的链表的头节点。

### 题解

```java
public ListNode deleteNode(ListNode head, int val) {
    ListNode preRoot = new ListNode();
    preRoot.next = head;

    ListNode pre = preRoot;
    ListNode cur = head;

    while(cur != null) {
        if (cur.val == val) {
            pre.next = cur.next;
            cur.next = null;
            cur = pre.next;
        } else {
            pre = cur;
            cur = cur.next;
        }
    }
    return preRoot.next;
}
```

## 27.复杂链表的复制

### 题目

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

### 题解

#### 用Map存新旧链表节点的映射关系，遍历两次

```java
public Node copyRandomList(Node head) {
    if (head == null) {
        return null;
    }

    Map<Node, Node> map = new HashMap<>();

    Node res = new Node(head.val);
    map.put(head, res);
    Node copyNode = res;
    Node cur1 = head.next;

    while (cur1 != null) {
        Node temp = new Node(cur1.val);
        copyNode.next = temp;
        map.put(cur1, temp);
        copyNode = copyNode.next;
        cur1 = cur1.next;
    }

    Node copyNode2 = res;
    Node cur2 = head;
    while (cur2 != null) {
        copyNode2.random = map.get(cur2.random);
        copyNode2 = copyNode2.next;
        cur2 = cur2.next;
    }
    return res;
}
```

#### 新节点接在旧节点后面

```java
public Node copyRandomList(Node head) {
    if(head == null) return null;
    Node cur = head;
    // 1. 复制各节点，并构建拼接链表
    while(cur != null) {
        Node tmp = new Node(cur.val);
        tmp.next = cur.next;
        cur.next = tmp;
        cur = tmp.next;
    }
    // 2. 构建各新节点的 random 指向
    cur = head;
    while(cur != null) {
        if(cur.random != null)
            cur.next.random = cur.random.next;
        cur = cur.next.next;
    }
    // 3. 拆分两链表
    cur = head.next;
    Node pre = head, res = head.next;
    while(cur.next != null) {
        pre.next = pre.next.next;
        cur.next = cur.next.next;
        pre = pre.next;
        cur = cur.next;
    }
    pre.next = null; // 单独处理原链表尾节点
    return res;      // 返回新链表头节点
}
```

## 28.最小的k个数

### 题目

输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

### 题解

#### 冒泡

```java
public int[] getLeastNumbers(int[] arr, int k) {
    int[] res = new int[k];
    for (int i = 0; i < k; i++) {
        for (int j = arr.length - 1; j > i; j--) {
            if (arr[j] < arr[j - 1]) {
                int temp = arr[j];
                arr[j] = arr[j - 1];
                arr[j - 1] = temp;
            }
        }
        res[i] = arr[i];
    }
    return res;
}
```

#### 优先级队列

```java
public int[] getLeastNumbers(int[] arr, int k) {
    int[] vec = new int[k];
    if (k == 0) { // 排除 0 的情况
        return vec;
    }
    PriorityQueue<Integer> queue = new PriorityQueue<Integer>(new Comparator<Integer>() {
        public int compare(Integer num1, Integer num2) {
            return num2 - num1;
        }
    });
    for (int i = 0; i < k; ++i) {
        queue.offer(arr[i]);
    }
    for (int i = k; i < arr.length; ++i) {
        if (queue.peek() > arr[i]) {
            queue.poll();
            queue.offer(arr[i]);
        }
    }
    for (int i = 0; i < k; ++i) {
        vec[i] = queue.poll();
    }
    return vec;
}
```

## 29.包含min函数的栈

### 题目

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

### 题解

#### 一个栈专门维护最小值

```java
class MinStack {

    private Deque<Integer> stack1 = new ArrayDeque();
    private Deque<Integer> stack2 = new ArrayDeque();

    /** initialize your data structure here. */
    public MinStack() {

    }

    public void push(int x) {
        stack1.push(x);
        if (stack2.isEmpty()) {
            stack2.push(x);
        } else {
            if (x < stack2.peek()) {
                stack2.push(x);
            } else {
                stack2.push(stack2.peek());
            }
        }
    }

    public void pop() {
        if (stack1.isEmpty()) {
            throw new RuntimeException("Stack is empty");
        }
        stack1.pop();
        stack2.pop();
    }

    public int top() {
        if (stack1.isEmpty()) {
            throw new RuntimeException("Stack is empty");
        }
        return stack1.peek();
    }

    public int min() {
        if (stack1.isEmpty()) {
            throw new RuntimeException("Stack is empty");
        }
        return stack2.peek();
    }
}
```

## 30.连续子数组的最大和

### 题目

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为O(n)。

### 题解

```java
public int maxSubArray(int[] nums) {
    int res = nums[0];
    int i = 1;
    int temp = res;

    while (i < nums.length) {
        if (temp <= 0) {
            temp = nums[i];
        } else {
            temp += nums[i];
        }
        if (temp > res) {
            res = temp;
        }
        i++;
    }

    return res;
}
```

## 31.正则表达式匹配

### 题目

请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

### 题解

```java
public boolean isMatch(String s, String p) {
    // dp[m][n]表示s的前m位和p的前n位匹配
    boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
    dp[0][0] = true;

    for (int i = 0; i <= s.length(); i++) {
        for (int j = 1; j <= p.length(); j++) {
            if (p.charAt(j - 1) == '*') {
                // *没匹配到
                dp[i][j] = dp[i][j - 2];

                // *至少匹配到一位
                if (isMatch(s, p, i, j - 1)) {
                    dp[i][j] = 
                        // 有可能上面dp[i][j] = dp[i][j - 2]已经是true了，不加会被dp[i - 1][j]覆盖
                        dp[i][j]
                        ||
                        // 如果*匹配了一位（第i位），去掉这个匹配到的第i位，s也能和p匹配
                        dp[i - 1][j];
                }
            } else {
                dp[i][j] = isMatch(s, p, i, j) && dp[i - 1][j - 1];
            }
        }
    }

    return dp[s.length()][p.length()];
}

private boolean isMatch(String s, String p, int i, int j) {
    if (i == 0) {
        return false;
    }
    return s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.';
}
```

## 32.数据流中的中位数

### 题目

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值

### 题解

#### 堆

```java
private PriorityQueue<Integer> queue1 = new PriorityQueue<>();
private PriorityQueue<Integer> queue2 = new PriorityQueue<>((a, b) -> b - a);

public MedianFinder() {

}

public void addNum(int num) {
    if (queue1.size() == queue2.size()) {
        queue2.add(num);
        queue1.add(queue2.poll());
    } else {
        queue1.add(num);
        queue2.add(queue1.poll());
    }
}

public double findMedian() {
    if (queue1.size() == queue2.size()) {
        return (queue1.peek() + queue2.peek()) / 2.0;
    } else {
        return queue1.peek();
    }
}
```

## 33.二叉搜索树与双向链表

### 题目

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

### 题解

```java
public Node treeToDoublyList(Node root) {
    if (root == null) {
        return null;
    }
    Deque<Node> stack = new ArrayDeque<>();
    Node res = null;
    Node pre = null;

    while (root != null || !stack.isEmpty()) {
        while(root != null) {
            stack.push(root);
            root = root.left;
        }
        root = stack.pop();

        if (res == null) {
            res = root;
        } else {
            pre.right = root;
            root.left = pre;
        }
        pre = root;
        root = root.right;
    }
    pre.right = res;
    res.left = pre;
    return res;
}
```

## 34.栈的压入、弹出序列

### 题目

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列

### 题解

#### 自己想的笨方法

```java
public boolean validateStackSequences(int[] pushed, int[] popped) {
    Deque<Integer> stack = new ArrayDeque<>();
    int i = 0, j = 0;

    while (j < popped.length) {
        while (i < pushed.length) {
            if (pushed[i] == popped[j]) {
                i++;
                j++;
            } else {
                if (!stack.isEmpty() && stack.peek() == popped[j]) {
                    stack.pop();
                    j++;
                } else {
                    stack.push(pushed[i]);
                    i++;
                }
            }
        }

        if (stack.isEmpty()) {
            return true;
        }

        if (stack.peek() == popped[j]) {
            stack.pop();
            j++;
        } else {
            break;
        }
    }
    return stack.isEmpty();
}
```

#### 答案

```java
public boolean validateStackSequences(int[] pushed, int[] popped) {
    Stack<Integer> stack = new Stack<>();
    int i = 0;
    for(int num : pushed) {
        stack.push(num); // num 入栈
        while(!stack.isEmpty() && stack.peek() == popped[i]) { // 循环判断与出栈
            stack.pop();
            i++;
        }
    }
    return stack.isEmpty();
}
```

## 35.序列化二叉树

### 题目

请实现两个函数，分别用来序列化和反序列化二叉树。

### 题解

```java
// Encodes a tree to a single string.
public String serialize(TreeNode root) {
    if (root == null) {
        return "[]";
    }
    StringBuilder res = new StringBuilder("[");
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);

    while (!queue.isEmpty()) {
        TreeNode temp = queue.poll();
        if (temp != null) {
            res.append(temp.val);
            res.append(",");
            queue.add(temp.left);
            queue.add(temp.right);
        } else {
            res.append("null,");
        }
    }
    res.deleteCharAt(res.length() - 1);
    res.append("]");
    return res.toString();
}

// Decodes your encoded data to tree.
public TreeNode deserialize(String data) {
    if ("".equals(data) || "[]".equals(data)) {
        return null;
    }

    String[] nodes = data.substring(1, data.length() - 1).split(",");
    TreeNode root = new TreeNode(Integer.valueOf(nodes[0]));
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);

    int start = 1;
    while (!queue.isEmpty()) {
        TreeNode temp = queue.poll();
        String cur = nodes[start];
        if (!"null".equals(cur)) {
            temp.left = new TreeNode(Integer.valueOf(cur));
            queue.add(temp.left);
        }
        cur = nodes[start + 1];
        if (!"null".equals(cur)) {
            temp.right = new TreeNode(Integer.valueOf(cur));
            queue.add(temp.right);
        }
        start += 2;
    }

    return root;
}
```

## 36.字符串的排列

### 题目

输入一个字符串，打印出该字符串中字符的所有排列。你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

### 题解

```java
private List<List<Character>> result = new ArrayList<>();

public String[] permutation(String s) {
    char[] array = s.toCharArray();
    List<Character> list = new ArrayList<>();
    for (char c : array) {
        list.add(c);
    }

    dfs(list, 0, array);

    Set<String> resSet = new HashSet(result.size());

    int i = 0;
    for (List<Character> charList : result) {
        StringBuilder sb = new StringBuilder();
        for (char c : charList) {
            sb.append(c);
        }
        resSet.add(sb.toString());
        i++;
    }

    String[] res = new String[resSet.size()];
    int j = 0;
    for (String str : resSet) {
        res[j++] = str;
    }
    return res;
}

private void dfs(List<Character> res, int index, char[] array) {
    if (index == array.length) {
        result.add(new ArrayList<>(res));
        return;
    }

    for (int i = index; i < array.length; i++) {
        if (i == index || array[i] != array[index]) {
            Collections.swap(res, index, i);
            dfs(res, index + 1, array);
            Collections.swap(res, index, i);
        }
    }
}
```

## 37.1-n的整数中1出现的次数

### 题目

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次

### 题解

#### 递归

```java
public int countDigitOne(int n) {
    if (n <= 0 ) {
        return 0;
    }
    String s = String.valueOf(n);
    int high = s.charAt(0) - '0';
    int power = (int) Math.pow(10, s.length() - 1);
    int last = n - high * power;

    if (high == 1) {
        return countDigitOne(power - 1) + last + 1 + countDigitOne(last);
    } else {
        return power + high * countDigitOne(power - 1) + countDigitOne(last);
    }
}
```

#### 看不懂的解法

```java
public int countDigitOne(int n) {
    int digit = 1, res = 0;
    int high = n / 10, cur = n % 10, low = 0;
    while(high != 0 || cur != 0) {
        if(cur == 0) res += high * digit;
        else if(cur == 1) res += high * digit + low + 1;
        else res += (high + 1) * digit;
        low += cur * digit;
        cur = high % 10;
        high /= 10;
        digit *= 10;
    }
    return res;
}
```

## 38.数组中出现次数超过一半的数字

### 题目

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

### 题解

#### 使用Map

```java
public int majorityElement(int[] nums) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int num : nums) {
        if (!map.containsKey(num)) {
            map.put(num, 1);
        } else {
            int tempCount = map.get(num) + 1;
            if (tempCount >= (nums.length + 1) / 2) {
                return num;
            } else {
                map.put(num, tempCount);
            }
        }
    }
    return nums[0];
}
```

#### 排序取中位数

```java
public int majorityElement(int[] nums) {
    Arrays.sort(nums);
    return nums[(nums.length - 1) / 2];
}
```

#### 摩尔投票法

```java
public int majorityElement(int[] nums) {
    int x = 0, votes = 0, count = 0;
    for(int num : nums){
        if(votes == 0) x = num;
        votes += num == x ? 1 : -1;
    }
    // 验证 x 是否为众数
    for(int num : nums)
        if(num == x) count++;
    return count > nums.length / 2 ? x : 0; // 当无众数时返回 0
}
```



## 39.二叉树层序遍历

### 题目

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

### 题解

```java
public int[] levelOrder(TreeNode root) {
    if (root == null) {
        return new int[0];
    }
    Queue<TreeNode> queue = new LinkedList<>();
    List<Integer> res = new ArrayList<>();

    queue.add(root);
    while (!queue.isEmpty()) {
        TreeNode temp = queue.poll();
        if (temp != null) {
            res.add(temp.val);
            queue.add(temp.left);
            queue.add(temp.right);
        }
    }

    int[] result = new int[res.size()];
    for (int i = 0; i < res.size(); i++) {
        result[i] = res.get(i);
    }
    return result;
}
```

## 40.二叉树层序遍历2

### 题目

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

###  题解

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) {
        return result;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);

    while (!queue.isEmpty()) {
        List<Integer> list = new ArrayList<>();
        int i = queue.size() - 1;
        while (i >= 0) {
            TreeNode tempNode = queue.poll();
            if (tempNode != null) {
                queue.add(tempNode.left);
                queue.add(tempNode.right);
                list.add(tempNode.val);
            }
            i--;
        }
        if (!list.isEmpty()) {
            result.add(list);
        }
    }
    return result;
}
```

## 41.从上到下打印二叉树3

### 题目

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

### 题解

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) {
        return result;
    }

    LinkedList<TreeNode> list = new LinkedList<>();
    list.add(root);

    boolean flag = true;
    while (!list.isEmpty()) {
        int i = list.size();
        List<Integer> tempList = new ArrayList<>();
        while (i > 0) {
            TreeNode temp;
            if (flag) {
                temp = list.poll();

                if (temp != null) {
                    tempList.add(temp.val);
                    list.add(temp.left);
                    list.add(temp.right);
                }
            } else {
                temp = list.removeLast();

                if (temp != null) {
                    tempList.add(temp.val);
                    list.addFirst(temp.right);
                    list.addFirst(temp.left);
                }
            }

            i--;
        }
        flag = !flag;
        if (!tempList.isEmpty()) {
            result.add(tempList);
        }
    }
    return result;
}
```

## 42.数字序列中某一位数字

### 题目

在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 *n* 个数字。

### 题解

```java
public int findNthDigit(int n) {
    long temp = 0, power = 9;
    int len = 1;
    while (true) {
        temp += len * power;

        if (temp > n) {
            break;
        } else {
            len++;
            power *= 10;
        }
    }
    int start = 0, tempLen = len;
    while (tempLen > 1) {
        start = start * 10 + 9;
        tempLen--;
    }

    int distance = n - (int) (temp - len * power);
    int consult =  distance / len;
    int mod = 0;
    if (len != 1) {
        mod = distance % len;
    }
    int num = start + consult;

    if (mod == 0) {
        return num % 10;
    } else {
        num = num + 1;
        return String.valueOf(num).charAt(mod - 1) - '0';
    }
}
```

## 43.二叉搜索树的后序遍历序列

### 题目

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同

### 题解

#### 递归

```java
public boolean verifyPostorder(int[] postorder) {
    if (postorder.length == 0 || postorder.length == 1) {
        return true;
    }
    return verifyPostorder(postorder, 0, postorder.length - 1);
}

private boolean verifyPostorder(int[] postorder, int start, int end) {
    if (start == end) {
        return true;
    }

    int rootVal = postorder[end];
    int rightVal = postorder[end - 1];

    if (rightVal < rootVal) {
        return verifyLeftValue(postorder, start, end - 1, rootVal) && verifyPostorder(postorder, start, end - 1);
    }

    int i = end - 2;
    // 右子树的全部值必须都大于root
    while (i >= start) {
        if (postorder[i] < rootVal) {
            break;
        }
        i--;
    }
    if (i < start) {
        return verifyPostorder(postorder, start, end - 1);
    }
    return verifyLeftValue(postorder, start, i, rootVal) && verifyPostorder(postorder, start, i) && verifyPostorder(postorder, i + 1, end - 1);
}

// 左子树的全部值都必须小于root
private boolean verifyLeftValue(int[] postorder, int start, int end, int rootVal) {
    for (int i = start; i <= end; i++) {
        if (postorder[i] > rootVal) {
            return false;
        }
    }
    return true;
}
```

## 44.第一个只出现一次的字符

### 题目

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

### 题解

```java
public char firstUniqChar(String s) {
    int[] stat = new int[26];
    char[] charArray = s.toCharArray();

    for (char c : charArray) {
        stat[c - 'a'] += 1;
    }
    for (char c : charArray) {
        if (stat[c - 'a'] == 1) {
            return c;
        }
    }

    return ' ';
}
```

## 45.和为s的两个数字

### 题目

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

### 题解

#### 传统Map解法

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();

    int[] result = new int[2];
    for (int i = 0; i < nums.length; i++) {
        Integer cur = map.get(target - nums[i]);
        if (cur != null) {
            result[0] = target - nums[i];
            result[1] = nums[i];
            return result;
        } else {
            map.put(nums[i], i);
        }
    }
    return result;
}
```

#### 双指针

```java
public int[] twoSum(int[] nums, int target) {
    int[] result = new int[2];
    int i = 0, j = nums.length - 1;

    while (i < j) {
        int temp = nums[i] + nums[j];
        if (temp == target) {
            result[0] = nums[i];
            result[1] = nums[j];
            return result;
        } else if (temp > target) {
            j--;
        } else {
            i++;
        }
    }
    return result;
}
```

## 46.二叉树的深度

### 题目

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

### 题解

```java
public int maxDepth(TreeNode root) {
    return root == null ? 0 : Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
}
```

## 47.二叉树中和为某一值的路径

### 题目

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

### 题解

#### dfs

```java
List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> pathSum(TreeNode root, int sum) {
    if (root != null) {
        List<Integer> list = new ArrayList<>();
        list.add(root.val);
        dfs(root, list, root.val, sum);
    }
    return result;
}

private void dfs(TreeNode cur, List<Integer> list, int curSum, int sum) {
    if (cur.left == null && cur.right == null) {
        if (curSum == sum) {
            result.add(new ArrayList<>(list));
        }
        return;
    }

    if (cur.left != null) {
        list.add(cur.left.val);
        dfs(cur.left, list, curSum + cur.left.val, sum);
        list.remove(list.size() - 1);
    }

    if (cur.right != null) {
        list.add(cur.right.val);
        dfs(cur.right, list, curSum + cur.right.val, sum);
        list.remove(list.size() - 1);
    }
}
```

## 48.数组中数字出现的次数1

### 题目

一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)

### 题解

#### 分组异或

```java
public int[] singleNumbers(int[] nums) {
    int k = 0;
    for (int n : nums) {
        k ^= n;
    }

    int mask = 1;
    while ((k & mask) == 0) {
        mask <<= 1;
    }

    int a = 0, b = 0;
    for (int n : nums) {
        if ((n & mask) == 0) {
            a ^= n;
        } else {
            b ^= n;
        }
    }
    int[] result = new int[2];
    result[0] = a;
    result[1] = b;
    return result;
}
```

## 49.数组中数字出现的次数2

### 题目

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字

### 题解

#### 统计二进制每位的1 的出现的次数

```java
public int singleNumber(int[] nums) {
    int[] res = new int[32];

    for (int num : nums) {
        int j = 31;
        while (num != 0) {
            res[j] += num & 1;
            j--;
            num >>= 1;
        }
    }

    int result = 0;
    for (int k : res) {
        result <<= 1;
        result += k % 3;
    }
    return result;
}
```

## 50.数组中的逆序对

### 题目

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

### 题解

```java
public int reversePairs(int[] nums) {
    int len = nums.length;
    if (len < 2) {
        return 0;
    }

    int[] temp = new int[len];
    return reversePairs(nums, 0, len - 1, temp);
}

private int reversePairs(int[] nums, int start, int end, int[] temp) {
    if (start == end) {
        return 0;
    }

    int mid = start + (end - start) / 2;
    int leftCount = reversePairs(nums, start, mid, temp);
    int rightCount = reversePairs(nums, mid + 1, end, temp);

    if (nums[mid] <= nums[mid + 1]) {
        return leftCount + rightCount;
    }
    return leftCount + rightCount + merge(nums, start, mid, end, temp);
}

private int merge(int[] nums, int start, int mid, int end, int[] temp) {
    for (int i = start; i <= end; i++) {
        temp[i] = nums[i];
    }
    int i = start, j = mid + 1;
    int count = 0;
    for (int k = start; k <= end; k++) {
        if (i == mid + 1) {
            nums[k] = temp[j++];
        } else if (j == end + 1) {
            nums[k] = temp[i++];
        } else if (temp[i] <= temp[j]) {
            nums[k] = temp[i++];
        } else {
            nums[k] = temp[j++];
            count += (mid - i + 1);
        }
    }
    return count;
}
```

## 51.把数组排成最小的数

### 题目

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个

### 题解

#### 比较(x + y)和(y + x)

```java
public String minNumber(int[] nums) {
    String[] strNums = new String[nums.length];
    for (int i = 0; i < nums.length; i++) {
        strNums[i] = String.valueOf(nums[i]);
    }

    Arrays.sort(strNums, (a, b) -> (a + b).compareTo(b + a));

    StringBuilder sb = new StringBuilder();
    for (String s : strNums) {
        sb.append(s);
    }
    return sb.toString();
}
```

## 52.和为s的连续正数序列

### 题目

输入一个正整数 `target` ，输出所有和为 `target` 的连续正整数序列（至少含有两个数）。序列内的数字由小到大排列，不同序列按照首个数字从小到大排列

### 题解

```java
public int[][] findContinuousSequence(int target) {
    List<List<Integer>> result = new ArrayList<>();
    int len = 2;
    int temp = 1;

    int row = 0;
    int col = 0;
    while (target - len >= temp) {
        if ((target - temp) % len == 0) {
            row++;
            col = len;

            List<Integer> list = new ArrayList<>();
            int k = (target - temp) / len;
            for (int i = 0; i <= len - 1; i++) {
                list.add(k + i);
            }
            result.add(list);
        }
        len++;
        temp = len * (len - 1) / 2;
    }

    int[][] res = new int[row][];
    for (int i = result.size() - 1; i >= 0 ; i--) {
        List<Integer> tempList = result.get(i);
        int index = result.size() - i - 1;
        res[index] = new int[tempList.size()];
        for (int j = 0; j < tempList.size(); j++) {
            res[index][j] = tempList.get(j);
        }
    }
    return res;
}
```

## 53.把数字翻译成字符串

### 题目

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

### 题解

#### dp

```java
public int translateNum(int num) {
    if (num < 10) {
        return 1;
    }

    String str = String.valueOf(num);
    int len = str.length();

    int[] dp = new int[len + 1];
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= len; i++) {
        if (canTranslate(str.charAt(i - 2), str.charAt(i - 1))) {
            dp[i] = dp[i - 1] + dp[i - 2];
        } else {
            dp[i] = dp[i - 1];
        }
    }
    return dp[len];
}

private boolean canTranslate(char c0, char c1) {
    return c0 == '1' || (c0 == '2' && c1 <= '5');
}
```

## 54.两个链表的第一个公共节点

### 题目

输入两个链表，找出它们的第一个公共节点。

### 题解

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) {
        return null;
    }
    int deepA = getDeep(headA);
    int deepB = getDeep(headB);

    if (deepA > deepB) {
        int i = deepA - deepB;
        while (i > 0) {
            headA = headA.next;
            i--;
        }
    } else if (deepA < deepB) {
        int i = deepB - deepA;
        while (i > 0) {
            headB = headB.next;
            i--;
        }
    }
    while (headA != null) {
        if (headA == headB) {
            return headA;
        }
        headA = headA.next;
        headB = headB.next;
    }
    return null;
}

private int getDeep(ListNode head) {
    int deep = 0;
    ListNode node = head;
    while (node != null) {
        deep++;
        node = node.next;
    }
    return deep;
}
```

