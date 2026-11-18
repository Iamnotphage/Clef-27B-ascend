
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_CLEF_GDN_CONV_QKV_H_
#define ACLNN_CLEF_GDN_CONV_QKV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnClefGdnConvQkvGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * eps : required
 * tileTokens : required
 * qOut : required
 * kOut : required
 * vOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClefGdnConvQkvGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    double eps,
    int64_t tileTokens,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnClefGdnConvQkv
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClefGdnConvQkv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
