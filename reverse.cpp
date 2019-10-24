class Solution {
public:
    int reverse(int x) {
        int max = 0x7fffffff;
        int min = 0x80000000;
        long res=0;  //使用长整型判断是否溢出
        while(x){
            res = res*10+x%10;
            x=x/10;
        }
        return (res>max || res<min)? 0:res;
    }
};