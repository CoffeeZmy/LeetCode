# LeetCode1-50

[TOC]



## 1.两数之和

### 题目

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

### 题解

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums.length == 0) {
            throw new IllegalArgumentException("parameters error");
        }
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int temp = target - nums[i];
			// map.containsKey()的时间复杂度？
            if (map.containsKey(temp)) {
                return new int[]{map.get(temp), i};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[]{};
    }
}
```

## 2.两数相加

### 题目

给出两个非空的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储一位 数字。如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。您可以假设除了数字 0 之外，这两个数都不会以 0 开头

### 题解

#### 普通遍历

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode result = new ListNode(0);
    ListNode currentNode = result;

    boolean hasCarry = false;
    while (l1 != null || l2 != null || hasCarry) {
        int temp = 0;

        if (l1 != null) {
            temp += l1.val;
            l1 = l1.next;
        }
        if (l2 != null) {
            temp += l2.val;
            l2 = l2.next;
        }
        if (hasCarry) {
            temp++;
        }
        hasCarry = temp >= 10;

        currentNode.next = new ListNode(temp % 10);
        currentNode = currentNode.next;
    }
    return result.next;
}
```

#### 递归

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    return addTwoNumbersImpl(l1, l2, 0);
}

private ListNode addTwoNumbersImpl(ListNode l1, ListNode l2, int carry) {
    if (l1 == null && l2 == null && carry == 0) {
        return null;
    }

    int val = carry;
    if (l1 != null) {
        val = val + l1.val;
        l1 = l1.next;
    }
    if (l2 != null) {
        val = val + l2.val;
        l2 = l2.next;
    }

    ListNode result = new ListNode(val % 10);
    result.next = addTwoNumbersImpl(l1, l2, val / 10);
    return result;
}
```

## 3.无重复字符的最长子串

### 题目

给定一个字符串，请你找出其中含有不重复字符的最长子串的长度

### 题解

#### 本人渣渣解法

```java
public int lengthOfLongestSubstring(String s) {
    int left = 0;
    int right = 0;
    int res = 0;
    Set<Character> set = new HashSet<>();

    while (right < s.length()) {
        char c = s.charAt(right);
        if (set.contains(c)) {
            // 第right个子字符与之前的字符重复
            // 从left + 1重新开始查找
            set.remove(s.charAt(left));
            left++;
            continue;
        } else {
            // 如果此时子串长度已大于之前的记录，则替换
            if ((right - left + 1) > res) {
                res = right - left + 1;
            }
        }
        right++;
        set.add(c);
    }
    return res;
}
```

#### LeetCode解法

```java
public int lengthOfLongestSubstring(String s) {
    int left = 0;
    int right = 0;
    int res = 0;
    Set<Character> set = new HashSet<>();

    while (left < s.length()) {
        if (left != 0) {
            // 移动左指针，直到把之前重复的那个值（此时right的值）去掉
            set.remove(s.charAt(left - 1));
        }
        // 找到从left开始的最大子串
        while (right < s.length() && !set.contains(s.charAt(right))) {
            if (right - left + 1 > res) {
                res = right - left + 1;
            }
            set.add(s.charAt(right));
            right++;
        }
        left++;
    }
    return res;
}
```

## 4.寻找两个两个正序数组的中位数（hard）

### 题目

给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。你可以假设 nums1 和 nums2 不会同时为空。

### 题解

#### O(m + n)复杂度解法

```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int m = nums1.length;
    int n = nums2.length;
    int i = 0, j = 0, res = 0;
    // 如果(m + n)为奇书，则第(m + n + 1)/2个数刚好是中位数
    // 如果是偶数，则中位数为第(m + n + 1)/2个数和下一个数的平均值
    int times = (m + n + 1) / 2;

    while (times > 0) {
        if (i == m) {
            res = nums2[j++];
        } else if (j == n) {
            res = nums1[i++];
        } else if (nums1[i] < nums2[j]) {
            res = nums1[i++];
        } else {
            res = nums2[j++];
        }
        times--;
    }
    if ((m + n) % 2 == 1) {
        return res;
    } else {
        // 需要找(m + n + 1)/2的下一个数
        int next = 0;
        if (i == m) {
            next = nums2[j];
        } else if (j == n) {
            next = nums1[i];
        } else if (nums1[i] < nums2[j]) {
            next = nums1[i];
        } else {
            next = nums2[j];
        }
        return (res + next) / 2.0;
    }
}
```

#### O(log(m+n))解法（LeetCode官方解答）

二分查找

```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int length1 = nums1.length, length2 = nums2.length;
    int totalLength = length1 + length2;
    if (totalLength % 2 == 1) {
        int midIndex = totalLength / 2;
        double median = getKthElement(nums1, nums2, midIndex + 1);
        return median;
    } else {
        int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
        double median = (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
        return median;
    }
}

public int getKthElement(int[] nums1, int[] nums2, int k) {
    /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

    int length1 = nums1.length, length2 = nums2.length;
    int index1 = 0, index2 = 0;
    int kthElement = 0;

    while (true) {
        // 边界情况
        if (index1 == length1) {
            return nums2[index2 + k - 1];
        }
        if (index2 == length2) {
            return nums1[index1 + k - 1];
        }
        if (k == 1) {
            return Math.min(nums1[index1], nums2[index2]);
        }

        // 正常情况
        int half = k / 2;
        int newIndex1 = Math.min(index1 + half, length1) - 1;
        int newIndex2 = Math.min(index2 + half, length2) - 1;
        int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
        if (pivot1 <= pivot2) {
            k -= (newIndex1 - index1 + 1);
            index1 = newIndex1 + 1;
        } else {
            k -= (newIndex2 - index2 + 1);
            index2 = newIndex2 + 1;
        }
    }
}
```

## 5.最长回文子串

### 题目

给定一个字符串s，找到s的最长的回文子串，假设s的最大长度为1000

### 题解

#### 暴力解法

```java
public String longestPalindrome(String s) {
    int length = s.length();
    // 最长回文数的长度
    int max = s.length();

    while (max > 0) {
        int left = 0, right = max - 1;
        while (right < length) {
            if (isPalindrome(s, left, right)) {
                return s.substring(left, right + 1);
            } else {
                left++;
                right++;
            }
        }
        max--;
    }
    return "";
}

private boolean isPalindrome(String s, int left, int right) {
    // 越界，结束递归
    if (left >= right) {
        return true;
    }
    // 如果s(i,j)为回文串，则一定有si == sj且s(i + 1, j - 1)为回文串
    return s.charAt(left) == s.charAt(right) && isPalindrome(s, left + 1, right - 1);
}
```

#### 动态规划

```java
public String longestPalindrome2(String s) {
    if (s.length() <= 1) {
        return s;
    }

    int length = s.length();
    // 使用数组把计算结果存起来
    boolean[][] dp = new boolean[length][length];
    char[] array = s.toCharArray();
    int maxLength = 1, begin = 0;

    for (int i = 0; i < length; i++) {
        dp[i][i] = true;
    }

    for (int j = 1; j < length; j++) {
        for (int i = 0; i < j; i++) {
            if (array[i] != array[j]) {
                dp[i][j] = false;
            } else {
                if (j - i < 3) {
                    dp[i][j] = true;
                } else {
                    // 如果s(i,j)为回文串，则一定有si == sj且s(i + 1, j - 1)为回文串
                    dp[i][j] = dp[i + 1][j - 1];
                }
            }
            if (dp[i][j] && j - i + 1 > maxLength) {
                maxLength = j - i + 1;
                begin = i;
            }
        }
    }
    return s.substring(begin, begin + maxLength);
}
```

## 6.Z字形变换

### 题目

将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。之后你的输出需要从左到右逐行读取，产生一个新的字符串

### 题解

#### 暴力解法

```java
public String convert(String s, int numRows) {
    if (s.length() == 0 || numRows == 1 || s.length() <= numRows) {
        return s;
    }
    StringBuilder[] stringBuilders = new StringBuilder[numRows];
    char[] charArray = s.toCharArray();

    // 每2n - 2个数为一组
    int group = numRows * 2 - 2;

    for (int i = 0; i < s.length(); i++) {
        int temp = i % group;
        // 记录每个位置所在的行数
        int row;
        if (temp % numRows == temp) {
            row = temp;
        } else {
            row = group - temp;
        }
        StringBuilder builder = stringBuilders[row];
        if (builder == null) {
            builder = new StringBuilder();
        }
        builder.append(charArray[i]);
        stringBuilders[row] = builder;
    }
    StringBuilder result = new StringBuilder();
    for (int row = 0; row < numRows; row++) {
        System.out.println(stringBuilders[row]);
        result.append(stringBuilders[row]);
    }
    return result.toString();
}
```

#### LeetCode解一

```java
public String convert(String s, int numRows) {
    if (numRows == 1) return s;

    List<StringBuilder> rows = new ArrayList<>();
    for (int i = 0; i < Math.min(numRows, s.length()); i++)
        rows.add(new StringBuilder());

    int curRow = 0;
    boolean goingDown = false;

    for (char c : s.toCharArray()) {
        rows.get(curRow).append(c);
        // 当curRow在第一行和最底行时会换行
        if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
        curRow += goingDown ? 1 : -1;
    }

    StringBuilder ret = new StringBuilder();
    for (StringBuilder row : rows) ret.append(row);
    return ret.toString();
}
```

#### LeetCode解二

```java
public String convert(String s, int numRows) {
	// 根据规律遍历
    if (numRows == 1) return s;

    StringBuilder ret = new StringBuilder();
    int n = s.length();
    int cycleLen = 2 * numRows - 2;

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j + i < n; j += cycleLen) {
            ret.append(s.charAt(j + i));
            if (i != 0 && i != numRows - 1 && j + cycleLen - i < n)
                ret.append(s.charAt(j + cycleLen - i));
        }
    }
    return ret.toString();
}
```

## 7.整数反转

### 题目

给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−2^31,  2^31 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。

### 题解

#### 暴力解法

```java
public int reverse(int x) {
    if (x == 0) {
        return 0;
    }

    long result = 0;
    while (x != 0) {
        int temp = x % 10;
        x = x / 10;
        result += result * 10 + temp;

        if (result > Integer.MAX_VALUE || result < Integer.MIN_VALUE) {
            return 0;
        }
    }
    return (int) result;
}
```

#### 优雅解法

```java
public int reverse(int x) {
    int result = 0;
    while (x != 0) {
        int temp = x % 10;
        x = x / 10;

        if (result > Integer.MAX_VALUE / 10 || result < Integer.MIN_VALUE / 10
            // 因为int最大值的个位数为7，最小值的个位数为-8
            || (result == Integer.MAX_VALUE / 10 && temp > 7)
            || (result == Integer.MIN_VALUE / 10 && temp < -8)) {
            return 0;
        }

        result = result * 10 + temp;
    }
    return result;
}
```

## 8.字符串转整数

### 题目

请你来实现一个 atoi 函数，使其能将字符串转换成整数，无法转换时返回0，超出int范围时分别返回最大值和最小值

### 题解

#### 臃肿解法

```java
public int myAtoi(String str) {
    if (str == null) {
        return 0;
    }
    char[] charArray = str.toCharArray();
    int left = -1, length = 0;
    // 是否为负数
    boolean isNegative = false;

    for (int i = 0; i < charArray.length; i++) {
        char cur = charArray[i];

        if ((cur == ' ') && left == -1) {
            continue;
        }

        if ((cur == '+' || cur == '-') && left == -1) {
            left = i + 1;
            isNegative = cur == '-';
        } else if (cur >= '0' && cur <= '9') {
            if (left == -1) {
                left = i;
            }
            if (cur == '0' && length == 0) {
                left++;
            } else {
                length++;

                if (length > 10) {
                    return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
                }
            }
        } else {
            break;
        }
    }
    if (length == 0) {
        return 0;
    }
    String result = str.substring(left, left + length);

    if (length == 10) {
        // 前9位
        int pre = Integer.valueOf(result.substring(0, 9));
        int last = Integer.valueOf(result.substring(9, 10));

        if (!isNegative && (pre > Integer.MAX_VALUE / 10 || (pre == Integer.MAX_VALUE / 10 && last > 7))) {
            return Integer.MAX_VALUE;
        }
        // 这里注意是大于等于，因为如果last等于8，则result有可能等于2^32，溢出
        if (isNegative && (pre > Integer.MAX_VALUE / 10 || (pre == Integer.MAX_VALUE / 10 && last >= 8))) {
            return Integer.MIN_VALUE;
        }
    }
    return isNegative ? (0 - Integer.valueOf(result)) : Integer.valueOf(result);
}
```

#### LeetCode解法（自动机 / 状态机）

这里定义4种状态：start、signed（确认符号）、in_number（添加数字）、end（结束）

![](https://assets.leetcode-cn.com/solution-static/8_fig1.PNG)

```java
class State {
    static final String START = "start";
    static final String SIGNED = "signed";
    static final String IN_NUMBER = "in_number";
    static final String END = "end";

    private Map<String, String[]> table = new HashMap<>();
    private String state = START;
    private boolean isNegative = false;
    private long value = 0;

    State() {
        // ' ' -> +/- -> number -> other
        table.put(START, new String[]{START, SIGNED, IN_NUMBER, END});
        table.put(SIGNED, new String[]{END, END, IN_NUMBER, END});
        table.put(IN_NUMBER, new String[]{END, END, IN_NUMBER, END});
        table.put(END, new String[]{END, END, END, END});
    }

    private int getCol(char c) {
        if (c == ' ') {
            return 0;
        } else if (c == '+' || c == '-') {
            return 1;
        } else if (c >= '0' && c <= '9') {
            return 2;
        }
        return 3;
    }

    private void process(char c) {
        state = table.get(state)[getCol(c)];

        if (state.equals(SIGNED)) {
            isNegative = c == '-';
        } else if (state.equals(IN_NUMBER)) {
            int num = c - '0';
            value = value * 10 + num;
            value = isNegative ? Math.min((long) Integer.MAX_VALUE + 1, value) : Math.min(Integer.MAX_VALUE, value);
        }
    }
}

	public int myAtoi(String str) {
        if (str == null) {
            return 0;
        }

        State state = new State();
        for (char c : str.toCharArray()) {
            state.process(c);
        }
        return state.isNegative ? (int) (state.value * -1) : (int) state.value;
    }
}
```

## 9.回文数

### 题目

判断一个整数是否为回文数

### 题解

#### 转字符串

```java
public boolean isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        String s = String.valueOf(x);
        return isStringPalindrome(s, 0, s.length() - 1);
    }

private boolean isStringPalindrome(String s, int left, int right) {
    if (left >= right) {
        return true;
    }
    return s.charAt(left) == s.charAt(right) && isStringPalindrome(s, left + 1, right - 1);
}
```

#### reverse一半

```java
public boolean isPalindrome(int x) {
    // 注意这里要排除非0的10的倍数
    if (x < 0 || (x % 10 == 0 && x != 0)) {
        return false;
    }

    int reverse = 0;
    while (x > reverse) {
        reverse = 10 * reverse + x % 10;
        x = x / 10;
    }
    return x == reverse || x == reverse / 10;
}
```

## 10.正则表达式匹配（hard）

### 题目

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

### 题解

#### 动态规划

```java
public boolean isMatch(String s, String p) {
    int m = s.length();
    int n = p.length();

    // dp[i][j]代表s的前i个字符与p的前j是否匹配
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;

    for (int i = 0; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            // 注意j最大值为n.length，所以是j - 1
            if (p.charAt(j - 1) == '*') {
                // p的最后两位没有用到
                dp[i][j] = dp[i][j - 2];

                // s的第i位与p的第j-1位匹配，如ab和a.* / ab*
                if (match(s, p, i, j - 1)) {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 2];
                }
            } else {
                // 如果两者最后一位都匹配，则需要前面也匹配
                if (match(s, p, i, j)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
    }
    return dp[m][n];
}

/**
 * s的第i个字符与p的第j个字符是否匹配
 */
private boolean match(String s, String p, int i, int j) {
    if (i == 0) {
        return false;
    }
    if (p.charAt(j - 1) == '.') {
        return true;
    }
    return s.charAt(i - 1) == p.charAt(j - 1);
}
```

## 11.盛最多水的容器

### 题目

给你n个非负整数a1、a2、a3... an，每个数代码坐标中的一个点(i, ai)，在坐标内画n条垂直线，垂直线i的两个端点分别为(i, ai)和(i, 0)。找出其中两条线，使得它们与x轴共同构成的容器可以容纳最多的水。

### 题解

#### 全遍历

```java
public int maxArea(int[] height) {
    int res = 0;
    for (int i = 0; i < height.length; i++) {
        for (int j = i + 1; j < height.length; j++) {
            int temp = (j - i) * Math.min(height[i], height[j]);
            if (temp > res) {
                res = temp;
            }
        }
    }
    return res;
}
```

#### 双指针

```java
public int maxArea(int[] height) {
    int left = 0, right = height.length - 1;

    int res = 0;
    while (left < right) {
        int temp;
        // 谁小谁移动指针
        if (height[left] < height[right]) {
            temp = height[left] * (right - left);
            left++;
        } else {
            temp = height[right] * (right - left);
            right--;
        }
        if (temp > res) {
            res = temp;
        }
    }
    return res;
}
```

## 12.整数转罗马数字

### 题目

罗马数字包含以下七种字符：I、V、X、L、C、D和M，分别对应数值1、5、10、50、100、500、1000。通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

- I可以放在V和X左边，表示4和9；
- X可以放在L和C左边，表示40和90；
- C可以放在D和M左边，表示400和900；

给定一个整数将其转为罗马数字，输入确保在1-3999范围内

### 题解

#### 顺思维解法

```java
public String intToRoman(int num) {
    Map<Integer, String> map = new HashMap<>();
    map.put(1, "I");
    map.put(5, "V");
    map.put(10, "X");
    map.put(50, "L");
    map.put(100, "C");
    map.put(500, "D");
    map.put(1000, "M");
    map.put(4, "IV");
    map.put(9, "IX");
    map.put(40, "XL");
    map.put(90, "XC");
    map.put(400, "CD");
    map.put(900, "CM");
    
    if (map.containsKey(num)) {
        return map.get(num);
    }

    int v = 1;
    StringBuilder res = new StringBuilder();
    while (num != 0) {
        int value = num % 10;
            
        if (value != 0) {
            String cur = map.get(value * v);
            if (cur == null) {
                cur = getCur(value, v, map);
            }
            res.insert(0, cur);
        }
        v = v * 10;
        num = num / 10;
    }
    return res.toString();
}

private String getCur(int value, int v, Map<Integer, String> map) {
    StringBuilder res = new StringBuilder();

    if (value > 5) {
        value = value % 5;
        res.append(map.get(5 * v));
    }
    while (value > 0) {
        res.append(map.get(v));
        value--;
    }
    return res.toString();
}
```

#### 贪心算法

为了表示一个给定的整数，我们寻找适合它的最大符号。我们减去它，然后寻找适合余数的最大符号，依此类推，直到余数为0。我们取出的每个符号都附加到输出的罗马数字字符串上。

```java
public String intToRoman(int num) {
    int[] intArray = new int[]{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    String[] strArray = new String[]{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

    StringBuilder res = new StringBuilder();
    for (int i = 0; i < intArray.length; i++) {
        int value = intArray[i];
        while (num >= value) {
            num = num - value;
            res.append(strArray[i]);
        }
    }
    return res.toString();
}
```

## 13.罗马数字转整数

### 题目

罗马数字转整数（规则见12）

### 题解

#### 顺思维解法

```java
public int romanToInt(String s) {
    Map<Character, Integer> map = new HashMap<>();
    map.put('I', 1);
    map.put('V', 5);
    map.put('X', 10);
    map.put('L', 50);
    map.put('C', 100);
    map.put('D', 500);
    map.put('M', 1000);

    int res = 0;
    char preValue = ' ';

    for (int i = 0; i < s.length(); i++) {
        char cur = s.charAt(i);
        int value = map.get(cur);
        if ((preValue == 'I' && (cur == 'V' || cur == 'X'))
            || (preValue == 'X' && (cur == 'L' || cur == 'C'))
            || (preValue == 'C' && (cur == 'D' || cur == 'M'))) {
            // 按规则处理特殊情况
            res = res - 2 * map.get(preValue) + value;
        } else {
            res = res + value;
        }
        preValue = cur;
    }
    return res;
}
```

#### 把特殊情况直接加入map

```java
public int romanToInt(String s) {
	Map<String, Integer> map = new HashMap<>();
	map.put("I", 1);
	map.put("IV", 4);
	map.put("V", 5);
	map.put("IX", 9);
	map.put("X", 10);
	map.put("XL", 40);
	map.put("L", 50);
	map.put("XC", 90);
	map.put("C", 100);
	map.put("CD", 400);
	map.put("D", 500);
	map.put("CM", 900);
	map.put("M", 1000);
	
	int ans = 0;
	for(int i = 0;i < s.length();) {
		if(i + 1 < s.length() && map.containsKey(s.substring(i, i+2))) {
			ans += map.get(s.substring(i, i+2));
			i += 2;
		} else {
			ans += map.get(s.substring(i, i+1));
			i ++;
		}
	}
	return ans;
}
```

## 14.最长公共前缀

### 题目

编写一个函数来查找字符串数组中的最长公共前缀，如果不存在，则返回空字符串，所有输入只包含小写字母。

### 题解

#### 遍历

```java
public String longestCommonPrefix(String[] strs) {
    if (strs.length == 0) {
        return "";
    }
    int i = 0;
    StringBuilder res = new StringBuilder();
    while (true) {
        char cur = ' ';
        for (String str : strs) {
            if (str.length() <= i) {
                return res.toString();
            } else if (cur == ' ') {
                cur = str.charAt(i);
            } else {
                if (str.charAt(i) != cur) {
                    return res.toString();
                }
            }
        }
        res.append(cur);
        i++;
    }
}
```

#### LeetCode解一

```java
public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0) {
        return "";
    }
    String prefix = strs[0];
    int count = strs.length;
    // 挨个获得相邻两个字符串的公共前缀
    for (int i = 1; i < count; i++) {
        prefix = longestCommonPrefix(prefix, strs[i]);
        if (prefix.length() == 0) {
            break;
        }
    }
    return prefix;
}

public String longestCommonPrefix(String str1, String str2) {
    int length = Math.min(str1.length(), str2.length());
    int index = 0;
    while (index < length && str1.charAt(index) == str2.charAt(index)) {
        index++;
    }
    return str1.substring(0, index);
}
```

## 15.三数之和

### 题目

给你一个包含三个整数的数组nums，判断nums中是否存在三个元素a，b，c，使得三数之和为0，请你找出所有满足条件且不重复的三元组

### 题解

#### 排序加左右指针

```java
public List<List<Integer>> threeSum(int[] nums) {
    if (nums.length < 3) {
        return Collections.emptyList();
    }
    Arrays.sort(nums);

    List<List<Integer>> result = new ArrayList<>();
    int i = 0;
    while (i < nums.length - 2) {
        if (i == 0 || nums[i] != nums[i - 1]) {
            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                if (left > i + 1 && nums[left] == nums[left - 1]) {
                    left++;
                } else if (right < nums.length - 1 && nums[right] == nums[right + 1]) {
                    right--;
                } else {
                    int temp = nums[i] + nums[left] + nums[right];
                    if (temp > 0) {
                        right--;
                    } else if (temp < 0) {
                        left++;
                    } else {
                        List<Integer> res = new ArrayList<>();
                        res.add(nums[i]);
                        res.add(nums[left]);
                        res.add(nums[right]);
                        result.add(res);
                        left++;
                        right--;
                    }
                }
            }
        }
        i++;
    }
    return result;
}
```

## 16.最接近的三数之和

### 题目

给你一个包含三个整数的数组nums和目标值target，判断nums中是否存在三个元素a，b，c，使得三数之和与target最接近，求这个最接近的值，假定只有唯一答案。

### 题解

#### 排序加双指针（同15）

```java
public int threeSumClosest(int[] nums, int target) {
    int i = 0, res = 10000, len = nums.length;
    Arrays.sort(nums);
    while (i < len - 2) {
        if (i == 0 || nums[i] != nums[i - 1]) {
            int left = i + 1, right = len - 1;
            while (left < right) {
                int temp = nums[i] + nums[left] + nums[right];

                if (temp - target == 0) {
                    return target;
                } else if (temp - target > 0) {
                    right--;
                } else {
                    left++;
                }
                if (Math.abs(temp - target) < Math.abs(res - target)) {
                    res = temp;
                }
            }
        }
        i++;
    }
    return res;
}
```

## 17.电话号码的字母组合

### 题目

给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。给出数据到字母的映射与电话按键相同

### 题解

#### 回溯法

```java
private Map<String, String> phoneNumberMap = new HashMap<>();

private List<String> result = new ArrayList<>();

public List<String> letterCombinations(String digits) {
    if (digits.length() > 0) {
        phoneNumberMap.put("2", "abc");
        phoneNumberMap.put("3", "def");
        phoneNumberMap.put("4", "ghi");
        phoneNumberMap.put("5", "jkl");
        phoneNumberMap.put("6", "mno");
        phoneNumberMap.put("7", "pqrs");
        phoneNumberMap.put("8", "tuv");
        phoneNumberMap.put("9", "wxyz");
        backTrack("", digits);
    }
    return result;
}

private void backTrack(String res, String digits) {
    if (digits.length() == 0) {
        result.add(res);
    } else {
        String letters = phoneNumberMap.get(digits.substring(0, 1));
        for (int i = 0; i < letters.length(); i++) {
            backTrack(res + letters.charAt(i), digits.substring(1));
        }
    }
}
```

## 18.四数之和

### 题目

给定一个包含n个整数的数组nums和一个目标值target，判断nums中是否存在四个元素，使得四数之和与target相等，找出所有满足条件且不重复的四元组

### 题解

#### 与15、16一样

```java
public List<List<Integer>> fourSum(int[] nums, int target) {
    List<List<Integer>> result = new ArrayList<>();
    Arrays.sort(nums);

    int i = 0, len = nums.length;
    while (i < len - 3) {
        if (i == 0 || nums[i] != nums[i - 1]) {
            int j = i + 1;
            while (j < len - 2) {
                if (j == i + 1 || nums[j] != nums[j - 1]) {
                    int left = j + 1, right = len - 1;
                    while (left < right) {
                        if (left != j + 1 && nums[left] == nums[left - 1]) {
                            left++;
                            continue;
                        }
                        if (right != len - 1 && nums[right] == nums[right + 1]) {
                            right--;
                            continue;
                        }
                        int temp = nums[i] + nums[j] + nums[left] + nums[right] - target;
                        if (temp > 0) {
                            right--;
                        } else if (temp < 0) {
                            left++;
                        } else {
                            List<Integer> res = new ArrayList<>();
                            res.add(nums[i]);
                            res.add(nums[j]);
                            res.add(nums[left]);
                            res.add(nums[right]);
                            result.add(res);
                            left++;
                            right--;
                        }
                    }
                }
                j++;
            }
        }
        i++;
    }
    return result;
}
```

## 19.删除链表的倒数第n个节点

### 题目

给定一个链表，删除链表的倒数第n个节点，并返回头节点

### 题解

#### 双指针

```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode preNode = null;
    ListNode firstNode = head;
    ListNode secondNode = head;

    int step = 0;
    while (firstNode != null) {
        firstNode = firstNode.next;
        step++;

        if (step > n) {
            preNode = secondNode;
            secondNode = secondNode.next;
        }
    }
    if (preNode != null) {
        preNode.next = secondNode.next;
    } else {
        return head.next;
    }
    return head;
}
```

## 20.有效的括号

### 题目

给定一个只包括小括号、中括号和大括号的字符串，判断字符串是否有效

### 题解

#### 使用栈

```java
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    int i = 0;
    char pre = ' ';
    while (i < s.length()) {
        char cur = s.charAt(i);
        if (isCharValid(pre, cur)) {
            // 如果匹配则取出
            stack.pop();
        } else {
            stack.push(cur);
        }
        if (!stack.isEmpty()) {
            pre = stack.peek();
        }
        i++;
    }
    return stack.isEmpty();
}

private boolean isCharValid(char left, char right) {
    return (left == '(' && right == ')') || (left == '[' && right == ']') || (left == '{' && right == '}');
}
```

#### LeetCode官方解法

```java
public boolean isValid(String s) {
    int n = s.length();
    if (n % 2 == 1) {
        return false;
    }

    Map<Character, Character> pairs = new HashMap<Character, Character>() {{
        put(')', '(');
        put(']', '[');
        put('}', '{');
    }};
    Stack<Character> stack = new Stack<Character>();
    for (int i = 0; i < n; i++) {
        char ch = s.charAt(i);
        if (pairs.containsKey(ch)) {
            if (stack.isEmpty() || stack.pop() != pairs.get(ch)) {
                return false;
            }
        } else {
            stack.push(ch);
        }
    }
    return stack.isEmpty();
}
```

#### LeetCode牛人解法

```java
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    for (char c : s.toCharArray()) {
        if (c == '(') {
            // push右括号，后面可以直接pop出来比较
            stack.push(')');
        } else if (c == '[') {
            stack.push(']');
        } else if (c == '{') {
            stack.push('}');
        } else if (stack.isEmpty() || c != stack.pop()) {
            // 如果要插入右括号时，栈为空，或者pop与c不匹配
            return false;
        }
    }
    return stack.isEmpty();
}
```

## 21.合并两个有序链表

### 题目

将两个升序链表合并为一个新的升序链表并返回

### 题解

#### 分支合并

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null) return l2;
    if (l2 == null) return l1;

    ListNode root;
    ListNode branchCur;

    if (l1.val <= l2.val) {
        root = l1;
        branchCur = l2;
    } else {
        root = l2;
        branchCur = l1;
    }
    ListNode preMainCur = root;
    ListNode mainCur = root.next;

    while (mainCur != null && branchCur != null) {
        if (mainCur.val <= branchCur.val) {
            preMainCur = mainCur;
            mainCur = mainCur.next;
        } else {
            ListNode temp = branchCur;
            branchCur = branchCur.next;
            preMainCur.next = temp;
            temp.next = mainCur;
            // 这里很容易漏，插入新节点后，preMainCur会变化
            preMainCur = temp;
        }
    }

    if (mainCur == null) {
        preMainCur.next = branchCur;
    }
    return root;
}
```

#### 递归

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null) {
        return l2;
    } else if (l2 == null) {
        return l1;
    } else if (l1.val <= l2.val) {
        l1.next = mergeTwoLists(l1.next, l2);
        return l1;
    } else {
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }
}
```

#### 引入一个preNode优化

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null) return l2;
    if (l2 == null) return l1;

    ListNode cur = new ListNode();
    ListNode pre = cur;

    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            cur.next = l1;
            l1 = l1.next;
        } else {
            cur.next = l2;
            l2 = l2.next;
        }
        cur = cur.next;
    }
    cur.next = l1 != null ? l1 : l2;
    return pre.next;
}
```

## 22.括号生成

### 题目

数字n代表生成括号的对数，请设计一个函数，用于生成所有可能的并且有效的括号组合

### 题解

#### 遍历

```java
private List<String> result = new ArrayList<>();

public List<String> generateParenthesis(int n) {
    generate("", n, n, 0);
    return result;
}

/**
     * @param str   结果
     * @param left  剩余左括号数
     * @param right 剩余右括号数
     * @param rest  未与右括号匹配的左括号数
     */
private void generate(String str, int left, int right, int rest) {
    if (left == 0 && right == 0) {
        result.add(str);
    } else if (left == 0) {
        // 右括号已用完
        generate(str + ")", left, right - 1, rest - 1);
    } else if (rest == 0) {
        // 下一个必须为左括号
        generate(str + "(", left - 1, right, rest + 1);
    } else {
        generate(str + "(", left - 1, right, rest + 1);
        generate(str + ")", left, right - 1, rest - 1);
    }
}
```

## 23.合并K个升序链表

### 题目

给你一个链表数组 ，每个链表都已经按升序排列，请你将所有链表合并到一个升序链表并返回

### 解法

#### 合并N个其实就是合并两个

```java
public ListNode mergeKLists(ListNode[] lists) {
    return mergeKListsImpl(lists, 0);
}

private ListNode mergeKListsImpl(ListNode[] lists, int start) {
    if (start == lists.length) {
        return null;
    }
    return mergeTwoListNode(lists[start], mergeKListsImpl(lists, start + 1));
}

private ListNode mergeTwoListNode(ListNode l1, ListNode l2) {
    if (l1 == null) return l2;
    if (l2 == null) return l1;

    ListNode cur = new ListNode();
    ListNode pre = cur;

    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            cur.next = l1;
            l1 = l1.next;
        } else {
            cur.next = l2;
            l2 = l2.next;
        }
        cur = cur.next;
    }
    cur.next = l1 != null ? l1 : l2;
    return pre.next;
}
```

#### 分治

```java
public ListNode mergeKLists(ListNode[] lists) {
    if (lists.length == 0) {
        return null;
    }
    return mergeListImpl(lists, 0, lists.length - 1);
}

private ListNode mergeListImpl(ListNode[] lists, int left, int right) {
    if (left == right) {
        return lists[left];
    }
    int mid = (left + right) / 2;
    return mergeTwoListNode(mergeListImpl(lists, left, mid), mergeListImpl(lists, mid + 1, right));
}

private ListNode mergeTwoListNode(ListNode l1, ListNode l2) {
    if (l1 == null) return l2;
    if (l2 == null) return l1;

    ListNode cur = new ListNode();
    ListNode pre = cur;

    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            cur.next = l1;
            l1 = l1.next;
        } else {
            cur.next = l2;
            l2 = l2.next;
        }
        cur = cur.next;
    }
    cur.next = l1 != null ? l1 : l2;
    return pre.next;
}
```

## 24.两两交换链表中的节点

### 题目

给定一个链表，两两交换其中**相邻**的节点，并返回交换后的链表

### 题解

#### 顺思维写

```java
public ListNode swapPairs(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }

    ListNode res = null;

    ListNode pre = new ListNode();
    while (head != null && head.next != null) {
        ListNode temp = head.next;
        head.next = temp.next;
        temp.next = head;
        pre.next = temp;
        pre = head;
        head = head.next;

        if (res == null) {
            res = temp;
        }
    }
    return res;
}
```

#### LeetCode题解

```java
public ListNode swapPairs(ListNode head) {
    ListNode pre = new ListNode(0);
    pre.next = head;
    ListNode temp = pre;
    while(temp.next != null && temp.next.next != null) {
        ListNode start = temp.next;
        ListNode end = temp.next.next;
        temp.next = end;
        start.next = end.next;
        end.next = start;
        temp = start;
    }
    return pre.next;
}
```

#### 递归

省略

## 25.K个一组翻转链表

### 题目

给你一个链表，每K个节点一组进行翻转，请你返回翻转后的链表，k为小于等于链表长度的正整数，如果节点总数不是k的整数倍，那么请将最后剩余的节点保持原有排序

### 题解

#### 遍历

```java
public ListNode reverseKGroup(ListNode head, int k) {
    if (head == null || head.next == null) {
        return head;
    }
    ListNode res = null;
    ListNode pre = new ListNode();
    pre.next = head;
    ListNode cur = pre;
    int i = k;
    while (cur.next != null) {
        cur = cur.next;
        i--;

        if (i == 0) {
            // 画图
            ListNode temp = cur.next;
            cur.next = null;
            reverseListNode(pre.next);

            if (res == null) {
                res = cur;
            }
            ListNode tempTail = pre.next;
            pre.next = cur;
            tempTail.next = temp;
            pre = tempTail;
            cur = pre;
            i = k;
        }
    }
    return res;
}

private ListNode reverseListNode(ListNode listNode) {
    if (listNode.next == null) {
        return listNode;
    }
    ListNode res = reverseListNode(listNode.next);
    listNode.next.next = listNode;
    listNode.next = null;
    return res;
}
```

## 26.删除排序数组中的重复项

### 题目

给定一个排序数组，你需要在原地删除重复出现的元素，使每个元素只出现一次，返回移除后数组的新长度

### 题解

#### 二分

```java
public int removeDuplicates(int[] nums) {
    if (nums.length == 0) {
        return 0;
    }
    return mergeDuplicates(nums, 0, nums.length - 1);
}

private int mergeDuplicates(int[] nums, int left, int right) {
    if (left == right) {
        return 1;
    }
    if (right - left == 1) {
        if (nums[left] == nums[right]) {
            return 1;
        } else {
            return 2;
        }
    }

    int mid = (left + right) / 2;
    int leftTotal = mergeDuplicates(nums, left, mid);
    int rightTotal = mergeDuplicates(nums, mid + 1, right);

    int leftIndex = left + leftTotal;
    int leftMaxValue = nums[leftIndex - 1];
    int i = 0;

    while (i < rightTotal) {
        if (leftMaxValue == nums[i + mid + 1]) {
            // 找到右边数组比左边最大值还要大的第一位
            i++;
            continue;
        }
        // 移动左右指针merge
        nums[leftIndex] = nums[i + mid + 1];
        leftIndex++;
        i++;
    }
    return leftIndex - left;
}
```

## 27.移除元素

### 题目

给你一个数组nums和一个值val，你需要原地移除所有数值等于val的元素，并返回移除后数组的新长度

### 题解

```java
public int removeElement(int[] nums, int val) {
    int i = 0, j = 0;
    while (j < nums.length) {
        if (nums[j] != val) {
            nums[i] = nums[j];
            i++;
        }
        j++;
    }
    return i;
}
```

## 28.实现strStr()

### 题目

给定一个haystack字符串和一个needle字符串，在haystack字符串中找出needle字符串出现的第一个位置（从0开始），如果不存在，则返回-1

### 题解

#### 暴力遍历

```java
public int strStr(String haystack, String needle) {
    if (haystack == null || needle == null) {
        return 0;
    }
    if (needle.length() == 0) {
        return 0;
    }
    if (haystack.length() == 0) {
        return -1;
    }

    char[] hayArray = haystack.toCharArray();
    char[] needleArray = needle.toCharArray();
    int i = 0, j = 0, k = 0;

    while (i <= hayArray.length - needleArray.length) {
        while (k < needleArray.length) {
            if (hayArray[i + j] == needleArray[k]) {
                k++;
                j++;
            } else {
                i++;
                j = 0;
                k = 0;
                break;
            }
            if (k == needleArray.length) {
                return i;
            }
        }
    }
    return -1;
}
```

## 29.两数相除

### 题目

给定两个整数，被除数dividend和除数divisor。将两数相除，要求不使用乘法、除法和mod运算符，返回商

### 题解

```java
public int divide(int dividend, int divisor) {
    if (dividend == Integer.MIN_VALUE && divisor == -1) {
        return Integer.MAX_VALUE;
    }
    if (dividend == divisor && divisor == Integer.MIN_VALUE) {
        return 1;
    }
    if (divisor == Integer.MIN_VALUE) {
        return 0;
    }
    boolean isPositive = (dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0);
    boolean overInt = dividend == Integer.MIN_VALUE;

    int resAbs = divideAbs(overInt ? Integer.MAX_VALUE : Math.abs(dividend), Math.abs(divisor), overInt);
    return isPositive ? resAbs : -resAbs;
}

private int divideAbs(int dividend, int divisor, boolean overInt) {
    if (dividend < divisor) {
        return 0;
    } else if (dividend == divisor) {
        return 1;
    } else {
        int divisorTemp = divisor;
        int res = 0;
        while (true) {
            res = res == 0 ? 1 : res + res;
            int temp = divisorTemp + divisorTemp;
            if (dividend > temp && temp > 0) {
                // temp < 0说明已溢出
                divisorTemp = temp;
            } else {
                break;
            }
        }
        int remainder = overInt ? (dividend - divisorTemp) + 1 : (dividend - divisorTemp);
        return res + divideAbs(remainder, divisor, false);
    }
}
```

## 30.串联所有单词的子串

### 题目

给定一个字符串s和一些长度相同的单词words。找出s中恰好可以由words中所有单词形成的子串的起始位置。注意子串要与words中的单词完全匹配，中间不能有其他字符，但不需要考虑words中单词串联的顺序。

### 题解

#### 暴力解法

```java
public List<Integer> findSubstring(String s, String[] words) {
    List<Integer> res = new ArrayList<>();

    if (words.length == 0 || s == null) {
        return res;
    }
    Map<String, Integer> wordsMap = new HashMap<>();
    for (String word : words) {
        // 记录words中每个单词出现次数
        wordsMap.put(word, wordsMap.getOrDefault(word, 0) + 1);
    }

    int l = words[0].length();
    int len = words.length * l;
    for (int i = 0; i <= s.length() - len; i++) {
        String subStr = s.substring(i, i + len);
        Map<String, Integer> subStrMap = new HashMap<>();
        for (int j = 0; j <= len - l; j = j + l) {
            String tempStr = subStr.substring(j, j + l);
            subStrMap.put(tempStr, subStrMap.getOrDefault(tempStr, 0) + 1);
        }
        if (subStrMap.equals(wordsMap)) {
            res.add(i);
        }
    }
    return res;
}
```

## 31.下一个排列

### 题目

实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）

### 题解

```java
public void nextPermutation(int[] nums) {
    int i = nums.length - 1;
    while (i > 0) {
        if (nums[i - 1] < nums[i]) {
            break;
        } else {
            i--;
        }
    }
    reverseArray(nums, i, nums.length - 1);

    if (i > 0) {
        for (int j = i; j < nums.length; j++) {
            if (nums[j] > nums[i - 1]) {
                int temp = nums[i - 1];
                nums[i - 1] = nums[j];
                nums[j] = temp;
                break;
            }
        }
    }
}

private void reverseArray(int[] nums, int left, int right) {
    while (left < right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
        left++;
        right--;
    }
}
```

## 32.最长有效括号（hard）

### 题目

给定一个只包含'('和')'的字符串，找出最长的包含有效括号的子串的长度

### 题解

#### 左右记数

```java
public int longestValidParentheses(String s) {
    int left = 0, right = 0, maxlength = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(') {
            left++;
        } else {
            right++;
        }
        if (left == right) {
            maxlength = Math.max(maxlength, 2 * right);
        } else if (right > left) {
            // 这里很关键
            left = right = 0;
        }
    }
    left = right = 0;
    for (int i = s.length() - 1; i >= 0; i--) {
        if (s.charAt(i) == '(') {
            left++;
        } else {
            right++;
        }
        if (left == right) {
            maxlength = Math.max(maxlength, 2 * left);
        } else if (left > right) {
            // 这里很关键
            left = right = 0;
        }
    }
    return maxlength;
}
```

#### 动态规划

```java
public int longestValidParentheses(String s) {
    int maxans = 0;
    int dp[] = new int[s.length()];
    for (int i = 1; i < s.length(); i++) {
        if (s.charAt(i) == ')') {
            if (s.charAt(i - 1) == '(') {
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
            }
            maxans = Math.max(maxans, dp[i]);
        }
    }
    return maxans;
}
```

## 33.搜索旋转排序数组

### 题目

假设按照升序的数组预先在某个点上进行了旋转，搜索一个给定的目标值，返回它的索引，要求O(logn)级别时间复杂度

### 题解

#### 递归

```java
public int search(int[] nums, int target) {
    if (nums.length == 0) {
        return -1;
    }
    return search(nums, target, 0, nums.length - 1);
}

private int search(int[] nums, int target, int left, int right) {
    if (left == right) {
        return nums[left] == target ? left : -1;
    }

    int mid = (left + right) / 2;
    if (nums[mid] >= nums[left]) {
        if (target >= nums[left] && target <= nums[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    } else {
        if (target >= nums[mid] && target <= nums[right]) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }
    return search(nums, target, left, right);
}
```

## 34.在排序数组中查找元素的第一个和最后一个位置

### 题目

给定一个按照升序排列的整数数组nms和一个目标值target，找出给定目标值在数组中的开始位置和结束位置。你的算法时间复杂度必须是 O(log n) 级别

### 题解

#### 二分法

```java
public int[] searchRange(int[] nums, int target) {
    int[] result = {-1, -1};

    int left = 0, right = nums.length - 1, mid;
    while (left <= right) {
        mid = (left + right) / 2;
        if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            result[0] = mid;
            result[1] = mid;

            int i = mid - 1;
            while (i >= 0 && nums[i] == target) {
                result[0] = i--;
            }
            int j = mid + 1;
            while (j < nums.length && nums[j] == target) {
                result[1] = j++;
            }
            break;
        }
    }
    return result;
}
```

## 35.搜索插入位置

### 题目

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

### 题解

```java
public int searchInsert(int[] nums, int target) {
    if (nums.length == 0) {
        return 0;
    } else if (nums.length == 1) {
        return target <= nums[0] ? 0 : 1;
    }
    boolean isAsc = nums[1] > nums[0];
    int left = 0, right = nums.length - 1, mid;
    while (left <= right) {
        mid = (left + right) / 2;
        if (nums[mid] == target) {
            return mid;
        } else if ((nums[mid] > target && isAsc) || (nums[mid] < target && !isAsc)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    int i = left - 1, j = left + 1;
    while (i >= 0 && ((nums[i] > target && isAsc) || (nums[i] < target && !isAsc))) {
        i--;
    }
    while (j < nums.length && ((nums[j] < target && isAsc) || (nums[j] > target && !isAsc))) {
        j++;
    }

    if (i != left - 1) {
        return i + 1;
    } else {
        return j - 1;
    }
}
```

## 36.有效的数独

### 题目

判断一个 9x9 的数独是否有效。只需要根据以下规则（1-9每行只能出现一次，1-9每列只能出现一次，1-9在每一个3*3宫内只能出现一次），验证已经填入的数字是否有效即可。

### 题解

#### 暴力

```java
public boolean isValidSudoku(char[][] board) {
    int rowLen = board.length, colLen = board[0].length;
    for (int i = 0; i < rowLen; i++) {
        if (!isGroupValid(board, i, i, 0, colLen - 1)) {
            return false;
        }
    }
    for (int j = 0; j < colLen; j++) {
        if (!isGroupValid(board, 0, rowLen - 1, j, j)) {
            return false;
        }
    }
    for (int k = 0; k < rowLen; k += 3) {
        for (int r = 0; r < colLen; r += 3) {
            if (!isGroupValid(board, k, k + 2, r, r + 2)) {
                return false;
            }
        }
    }
    return true;
}

private boolean isGroupValid(char[][] board, int rowLeft, int rowRight, int colLeft, int colRight) {
    boolean[] positions = new boolean[9];

    int i = rowLeft, j = colLeft;
    while (i <= rowRight) {
        while (j <= colRight) {
            char cur = board[i][j];
            if (cur != '.') {
                if (positions[cur - '1']) {
                    return false;
                } else {
                    positions[cur - '1'] = true;
                }
            }
            j++;
        }
        i++;
        j = colLeft;
    }
    return true;
}
```

## 37.解数独（hard）

### 题目

编写一个程序，通过已填充的空格来解决数独问题，规则同36

### 题解

```java
private int len = 9;
private boolean isSolved = false;
private boolean[][] rowPositions = new boolean[len][len];
private boolean[][] colPositions = new boolean[len][len];
private boolean[][] groupPositions = new boolean[len][len];

public void solveSudoku(char[][] board) {
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            char cur = board[i][j];
            int groupIndex = i / 3 + (j / 3) * 3;
            if (cur != '.') {
                int index = cur - '1';
                rowPositions[i][index] = true;
                colPositions[j][index] = true;
                groupPositions[groupIndex][index] = true;
            }
        }
    }
    backTrace(0, 0, board);
}

private boolean canPut(int row, int col, char c) {
    int index = c - '1';
    int groupIndex = row / 3 + (col / 3) * 3;
    return !rowPositions[row][index] && !colPositions[col][index] && !groupPositions[groupIndex][index];
}

private void backTrace(int row, int col, char[][] board) {
    char cur = board[row][col];

    if (cur == '.') {
        for (char c = '1'; c <= '9'; c++) {
            if (canPut(row, col, c)) {
                put(row, col, board, c);
                putNexts(row, col, board);

                if (!isSolved) {
                    remove(row, col, board);
                }
            }
        }
    } else {
        putNexts(row, col, board);
    }
}

private void putNexts(int row, int col, char[][] board) {
    if (row == len - 1 && col == len - 1) {
        isSolved = true;
    } else {
        if (col == len - 1) {
            backTrace(row + 1, 0, board);
        } else {
            backTrace(row, col + 1, board);
        }
    }
}

private void put(int row, int col, char[][] board, char c) {
    int index = c - '1';
    int groupIndex = row / 3 + (col / 3) * 3;

    board[row][col] = c;
    rowPositions[row][index] = true;
    colPositions[col][index] = true;
    groupPositions[groupIndex][index] = true;
}

private void remove(int row, int col, char[][] board) {
    int index = board[row][col] - '1';
    int groupIndex = row / 3 + (col / 3) * 3;

    board[row][col] = '.';
    rowPositions[row][index] = false;
    colPositions[col][index] = false;
    groupPositions[groupIndex][index] = false;
}
```

## 38.外观数列

### 题目

给定一个正整数 n（1 ≤ n ≤ 30），输出外观数列的第 n 项

### 题解

```java
public String countAndSay(int n) {
    if (n == 1) {
        return "1";
    }
    return getNext(countAndSay(n - 1));
}

private String getNext(String str) {
    int p = 0, i = 1, curCount = 1;
    StringBuilder res = new StringBuilder();

    while (p < str.length() && i < str.length()) {
        char cur = str.charAt(p);

        if (str.charAt(i) == cur) {
            curCount++;
        } else {
            res.append(curCount);
            res.append(cur);
            p = i;
            curCount = 1;
        }
        i++;
    }
    res.append(curCount);
    res.append(str.charAt(str.length() - 1));
    return res.toString();
}
```

## 39.组合总和

### 题目

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的数字可以无限制重复被选取。

### 题解

#### DFS（深度优先遍历）

```java
List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> combinationSum(int[] candidates, int target) {
    dfs(new ArrayList<>(), candidates, 0, target);
    return result;
}


private void dfs(List<Integer> res, int[] candidates, int index, int target) {
    if (index == candidates.length) {
        return;
    }
    if (target == 0) {
        result.add(new ArrayList<>(res));
        return;
    }
    // 假设当前值无法获得解，则
    dfs(res, candidates, index + 1, target);

    // 假设当前值能获得解，则
    if (candidates[index] <= target) {
        res.add(candidates[index]);
        dfs(res, candidates, index, target - candidates[index]);
        // 当前值无法获得解，需remove
        res.remove(res.size() - 1);
    }
}
```

## 40.组合总和2

### 题目

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能使用一次

### 题解

#### 同39

```java
List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    Arrays.sort(candidates);
    dfs(new ArrayList<>(), candidates, 0, target);
    return result;
}

private void dfs(List<Integer> res, int[] candidates, int index, int target) {
    if (target == 0) {
        result.add(new ArrayList<>(res));
        return;
    }

    if (index == candidates.length || candidates[index] > target) {
        return;
    }

    // 假设当前值无法获得解，则
    int i = index + 1;
    while (i < candidates.length && candidates[i] == candidates[index]) {
        i++;
    }
    dfs(res, candidates, i, target);

    // 假设当前值能获得解，则
    res.add(candidates[index]);
    dfs(res, candidates, index + 1, target - candidates[index]);
    // 当前值无法获得解，需remove
    res.remove(res.size() - 1);
}
```

## 41.缺失的第一个正数

### 题目

给一个未排序的整数数组，请你找出其中没有出现的最小正整数

### 题目

#### 原地哈希

```java
public int firstMissingPositive(int[] nums) {
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] <= 0) {
            nums[i] = nums.length + 1;
        }
    }

    for (int i = 0; i < nums.length; i++) {
        int temp = Math.abs(nums[i]);
        if (temp >= 1 && temp <= nums.length) {
            nums[temp - 1] = -Math.abs(nums[temp - 1]);
        }
    }

    for (int i = 0; i < nums.length; i++) {
        if (nums[i] > 0) {
            return i + 1;
        }
    }
    return nums.length + 1;
}
```

## 42.接雨水

### 题目

给定n个非负整数表示宽度为1的柱子的高度图，计算按此排列的柱子，下雨后能接多少水

### 题解

#### 暴力法

```java
public int trap(int[] height) {
    int res = 0;
    int leftMax, rightMax;

    // 计算1至n - 2的每个位置能储存的雨水
    for (int i = 1; i < height.length - 1; i++) {
        leftMax = 0;
        rightMax = 0;
        for (int j = i; j >= 0; j--) {
            leftMax = Math.max(height[j], leftMax);
        }
        for (int j = i; j < height.length; j++) {
            rightMax = Math.max(height[j], rightMax);
        }
        res += Math.min(leftMax, rightMax) - height[i];
    }
    return res;
}
```

#### 双指针法

```java
public int trap(int[] height) {
    int res = 0;
    int left = 0, right = height.length - 1;
    int leftMax = 0, rightMax = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] > leftMax) {
                leftMax = height[left];
            } else {
                res += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] > rightMax) {
                rightMax = height[right];
            } else {
                res += rightMax - height[right];
            }
            right--;
        }

    }
    return res;
}
```

## 43.字符串相乘

### 题目

给定两个以字符串形式表示的非负整数num1和num2，返回它们的乘积，也用字符串表示。num1和num2的长度小于110，均不以0开头（除非是0本身），不能使用标准库

### 题解

#### 暴力法

```java
/**
 *	乘法转加法
 */
public String multiply(String num1, String num2) {
    if (num1.equals("0") || num2.equals("0")) {
        return "0";
    }

    int len2 = num2.length();

    String res = "";
    for (int i = len2 - 1; i >= 0; i--) {
        String tempRes = "";
        int times = num2.charAt(i) - '0';
        while (times > 0) {
            tempRes = add(tempRes, num1);
            times--;
        }
        int zeroNum = len2 - 1 - i;
        while (zeroNum > 0) {
            tempRes = tempRes + "0";
            zeroNum--;
        }
        res = add(res, tempRes);
    }

    return res;
}

private String add(String num1, String num2) {
    if (num2.equals("0")) {
        return num1;
    }
    if (num1.equals("0")) {
        return num2;
    }

    int l1 = num1.length(), l2 = num2.length();
    int minLen = Math.min(l1, l2);

    int i = 1, carry = 0;
    StringBuilder res = new StringBuilder();

    while (i <= minLen) {
        int index1 = num1.length() - i;
        int index2 = num2.length() - i;

        char c1 = num1.charAt(index1);
        char c2 = num2.charAt(index2);

        int cur = (c1 - '0') + (c2 - '0') + carry;

        carry = cur / 10;
        res.insert(0, cur % 10);
        i++;
    }
    if (num1.length() > minLen) {
        return res.insert(0, add(num1.substring(0, num1.length() - minLen), String.valueOf(carry))).toString();
    } else {
        return res.insert(0, add(num2.substring(0, num2.length() - minLen), String.valueOf(carry))).toString();
    }
}
```

#### 优化addString和multiply

```java
public String multiply(String num1, String num2) {
    if (num1.equals("0") || num2.equals("0")) {
        return "0";
    }
    String ans = "0";
    int m = num1.length(), n = num2.length();
    for (int i = n - 1; i >= 0; i--) {
        StringBuffer curr = new StringBuffer();
        int add = 0;
        for (int j = n - 1; j > i; j--) {
            curr.append(0);
        }
        int y = num2.charAt(i) - '0';
        for (int j = m - 1; j >= 0; j--) {
            int x = num1.charAt(j) - '0';
            int product = x * y + add;
            curr.append(product % 10);
            add = product / 10;
        }
        if (add != 0) {
            curr.append(add % 10);
        }
        ans = addStrings(ans, curr.reverse().toString());
    }
    return ans;
}

public String addStrings(String num1, String num2) {
    int i = num1.length() - 1, j = num2.length() - 1, add = 0;
    StringBuffer ans = new StringBuffer();
    while (i >= 0 || j >= 0 || add != 0) {
        int x = i >= 0 ? num1.charAt(i) - '0' : 0;
        int y = j >= 0 ? num2.charAt(j) - '0' : 0;
        int result = x + y + add;
        ans.append(result % 10);
        add = result / 10;
        i--;
        j--;
    }
    ans.reverse();
    return ans.toString();
}
```

#### 做乘法

```java
public String multiply(String num1, String num2) {
    if (num1.equals("0") || num2.equals("0")) {
        return "0";
    }
    int m = num1.length(), n = num2.length();
    int[] ansArr = new int[m + n];
    for (int i = m - 1; i >= 0; i--) {
        int x = num1.charAt(i) - '0';
        for (int j = n - 1; j >= 0; j--) {
            int y = num2.charAt(j) - '0';
            ansArr[i + j + 1] += x * y;
        }
    }
    for (int i = m + n - 1; i > 0; i--) {
        ansArr[i - 1] += ansArr[i] / 10;
        ansArr[i] %= 10;
    }
    int index = ansArr[0] == 0 ? 1 : 0;
    StringBuffer ans = new StringBuffer();
    while (index < m + n) {
        ans.append(ansArr[index]);
        index++;
    }
    return ans.toString();
}
```

## 44.通配符匹配

### 题目

给定一个字符串s和字符模式p，实现一个支持'?'和'*'的通配符匹配，?可以匹配任何单个字符，'*'可以匹配任意字符串（包括空字符）

### 题解

#### DP

```java
public boolean isMatch(String s, String p) {
    int sLen = s.length();
    int pLen = p.length();

    // dp[i][j]表示s的前i个字符与p的前j个字符匹配
    boolean[][] dp = new boolean[sLen + 1][pLen + 1];

    dp[0][0] = true;

    // 覆盖p全为*的情况
    for (int i = 1; i <= pLen; i++) {
        if (p.charAt(i - 1) == '*') {
            dp[0][i] = true;
        } else {
            break;
        }
    }

    for (int i = 1; i <= sLen; i++) {
        for (int j = 1; j <= pLen; j++) {
            if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                dp[i][j] = dp[i - 1][j - 1];
            } else if (p.charAt(j - 1) == '*') {
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            }
        }
    }
    return dp[sLen][pLen];
}
```

## 45.跳跃游戏二

### 题目

给定一个非负整数数组，你最初位于数组的第一个位置，数组中的每个元素代表你在该位置可以跳跃的最大长度。你的目标是使用最少的跳跃次数到达数组的最后一个位置。

### 题解

#### 从后往前

```java
public int jump(int[] nums) {
    if (nums.length == 0 || nums.length == 1) {
        return 0;
    }
    int l = nums.length - 1;
    int res = 0;

    while (l > 0) {
        int i = l - 1;
        int pre = i;
        while (i >= 0) {
            if (nums[i] >= l - i) {
                pre = i;
            }
            i--;
        }
        l = pre;
        res++;
    }
    return res;
}
```

#### 从前往后

```java
public int jump(int[] nums) {
    int length = nums.length;
    int end = 0;
    int maxPosition = 0; 
    int steps = 0;
    for (int i = 0; i < length - 1; i++) {
        maxPosition = Math.max(maxPosition, i + nums[i]); 
        if (i == end) {
            end = maxPosition;
            steps++;
        }
    }
    return steps;
}
```

## 46.全排列

### 题解

给定一个没有重复数字的序列，返回所有可能的全排列

### 题解

#### 个人解

```java
List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> permute(int[] nums) {
    group(new ArrayList<>(), nums, 0);
    return result;
}

private void group(List<Integer> list, int[] nums, int index) {
    if (index == nums.length) {
        result.add(new ArrayList<>(list));
        return;
    }
    int size = list.size();
    for (int i = 0; i < size + 1; i++) {
        list.add(i, nums[index]);
        group(list, nums, index + 1);
        list.remove(i);
    }
}
```

#### dfs

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> res = new LinkedList();
    ArrayList<Integer> output = new ArrayList<Integer>();
    for (int num : nums)
        output.add(num);

    int n = nums.length;
    backtrack(n, output, res, 0);
    return res;
}

public void backtrack(int n, ArrayList<Integer> output, List<List<Integer>> res, int first) {
   // 定义递归函数 backtrack(first, output) 表示从左往右填到第first个位置，当前排列为output
    if (first == n) {
        // 所有数都填完了
        res.add(new ArrayList<Integer>(output));
    }
        
    for (int i = first; i < n; i++) {
        // 动态维护数组
        Collections.swap(output, first, i);
        // 继续递归填下一个数
        backtrack(n, output, res, first + 1);
        // 撤销操作
        Collections.swap(output, first, i);
    }
}
```

## 47.全排列2

### 题目

给定一个可包含重复数字的序列，返回所有不重复的全排列。

### 题解

#### dfs

```java
// nums每个位置的数字是否已填入
boolean[] vls;
List<List<Integer>> res = new ArrayList<>();

public List<List<Integer>> permuteUnique(int[] nums) {
    vls = new boolean[nums.length];

    List<Integer> output = new ArrayList<>();
    Arrays.sort(nums);
    backTrace(output, 0, nums);
    return res;
}

private void backTrace(List<Integer> output, int first, int[] nums) {
    if (first == nums.length) {
        res.add(new ArrayList<>(output));
    }
    for (int i = 0; i < nums.length; i++) {
        if (vls[i] ||
            // 加!vls[i-1]是为了剪枝
            (i > 0 && nums[i - 1] == nums[i] && !vls[i - 1])) {
            continue;
        }
        output.add(nums[i]);
        vls[i] = true;
        backTrace(output, first + 1, nums);
        output.remove(first);
        vls[i] = false;
    }
}
```

## 48.旋转图像

### 题目

给定一个n x n的二维矩阵表示一个图像，将图像顺时针旋转90度

### 题解

#### 从外到内

```java
public void rotate(int[][] matrix) {
        int n = matrix.length;

        int loop = 0;
        while (loop < n / 2) {
            int i = loop;

            for (int j = i; j < n - 1 - loop; j++) {
                int rotateRow = i;
                int rotateCol = j;
                int value = matrix[rotateRow][rotateCol];

                while (true) {
                    int t = rotateRow;
                    rotateRow = rotateCol;
                    rotateCol = n - 1 - t;

                    int temp = matrix[rotateRow][rotateCol];
                    matrix[rotateRow][rotateCol] = value;
                    value = temp;

                    if (rotateRow == i && rotateCol == j) {
                        break;
                    }
                }
            }
            loop++;
        }
    }
```

#### 转职加翻转

```java
public void rotate(int[][] matrix) {
    int n = matrix.length;

    // transpose matrix
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int tmp = matrix[j][i];
            matrix[j][i] = matrix[i][j];
            matrix[i][j] = tmp;
        }
    }
    // reverse each row
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n / 2; j++) {
            int tmp = matrix[i][j];
            matrix[i][j] = matrix[i][n - j - 1];
            matrix[i][n - j - 1] = tmp;
        }
    }
}
```

## 49.字母异位词分组

### 题目

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串

### 题解

#### 笨比方法（会超出时间限制）

```java
public List<List<String>> groupAnagrams(String[] strs) {
    // key是每个异位词组合的第一个，value是组合
    Map<String, List<String>> map = new HashMap<>();
    // 与map同key，value是该词每个字符的数量
    Map<String, Map<Character, Integer>> keyMap = new HashMap<>();

    for (String str : strs) {
        String key = null;
        for (Map.Entry<String, Map<Character, Integer>> entry : keyMap.entrySet()) {
            if (str.length() == entry.getKey().length() && isHeterotopicWords(entry.getValue(), str)) {
                key = entry.getKey();
                break;
            }
        }
        if (key == null) {
            // 该单词是该异位词的第一个
            keyMap.put(str, generateCharMap(str));

            List<String> list = new ArrayList<>();
            list.add(str);
            map.put(str, list);
        } else {
            map.get(key).add(str);
        }
    }
    return new ArrayList<>(map.values());
}

private Map<Character, Integer> generateCharMap(String word) {
    Map<Character, Integer> map = new HashMap<>();
    for (char c : word.toCharArray()) {
        if (map.containsKey(c)) {
            map.put(c, map.get(c) + 1);
        } else {
            map.put(c, 1);
        }
    }
    return map;
}

private boolean isHeterotopicWords(Map<Character, Integer> map, String word) {
    Map<Character, Integer> tempMap = new HashMap<>(map);

    for (char c : word.toCharArray()) {
        if (tempMap.containsKey(c)) {
            if (tempMap.get(c) == 0) {
                return false;
            } else {
                tempMap.put(c, tempMap.get(c) - 1);
            }
        } else {
            return false;
        }
    }
    return true;
}
```

#### 笨比方式的基础上把key按字母顺序排个序

```java
public List<List<String>> groupAnagrams(String[] strs) {
    // key是每个异位词组合的第一个，value是组合
    Map<String, List<String>> map = new HashMap<>();

    for (String str : strs) {
        char[] strArray = str.toCharArray();
        Arrays.sort(strArray);
        String key = String.valueOf(strArray);

        if (map.containsKey(key)) {
            map.get(key).add(str);
        } else {
            List<String> list = new ArrayList<>();
            list.add(str);
            map.put(key, list);
        }
    }
    return new ArrayList<>(map.values());
}
```

## 50.Power(x,n)

### 题目

实现power(x,n)，即计算x的n次幂函数，-100 < x < 100，n为32位有符号整数

### 题解

```java
public double myPow(double x, int n) {
    if (n == 0) {
        return 1.0;
    }

    if (n == Integer.MIN_VALUE) {
        return myPow(x, n + 1) / x;
    }

    boolean isPositive = n > 0;
    int power = isPositive ? n : -n;
    double result;

    double temp = myPow(x, power / 2);
    if (power % 2 == 0) {
        result = temp * temp;
    } else {
        result = x * temp * temp;
    }
    return isPositive ? result : 1 / result;
}
```

