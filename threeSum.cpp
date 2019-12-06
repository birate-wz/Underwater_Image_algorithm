class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        int len = nums.size();
        if(len<3) return res;
        sort(nums.begin(),nums.end());
        for(int i=0;i<len;i++){
           if(nums[i]>0) break;
           if(i > 0 && nums[i] == nums[i-1]) continue;  //去重
           int L = i+1;
           int R = len-1;
           while(L<R){
              int sum = nums[i]+nums[L]+nums[R];
              if(sum==0){
                  vector<int> temp;
                  temp.push_back(nums[i]);
                  temp.push_back(nums[L]);
                  temp.push_back(nums[R]);
                  res.push_back(temp);
                  while(L<R && nums[L]==nums[L+1]) L++;
                  while(L<R && nums[R]==nums[R-1]) R--;
                  L++;
                  R--; 
              } 
              else if(sum<0) L++;
              else if(sum>0) R--;
           }  
        }
        return res;
    }
};