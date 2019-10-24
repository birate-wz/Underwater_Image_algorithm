/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode newhead(0);
        ListNode *p = &newhead;
        int flag=0;
        while(l1||l2){
            int x = (l1!=NULL) ? l1->val:0;
            int y = (l2!=NULL) ? l2->val:0;
            int temp = x + y + flag;
            flag = temp/10;
            p->next = new ListNode(temp%10);
            p = p->next;
            if(l1) l1=l1->next;
            if(l2) l2=l2->next;
        }
        if(flag>0)
            p->next = new ListNode(flag);
        return newhead.next;
    }
};