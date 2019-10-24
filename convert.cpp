class Solution {
public:
    string convert(string s, int numRows) {
        if(numRows==1) return s;

        vector<string> rows(min(numRows,int(s.size())));  //存储每一行的值
        int cursum=0;
        int flag=-1;
        for(char c : s){
            rows[cursum] += c;
            if(cursum==0 || cursum == numRows-1) flag = -flag;
            cursum += flag;
        }
        string ret;
        for(string res:rows)
            ret+=res;
        return ret;
    }
};