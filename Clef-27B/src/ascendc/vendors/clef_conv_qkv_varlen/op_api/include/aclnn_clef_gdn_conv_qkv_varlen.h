
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_CLEF_GDN_CONV_QKV_VARLEN_H_
#define ACLNN_CLEF_GDN_CONV_QKV_VARLEN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnClefGdnConvQkvVarlenGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * cuSeqlens : required
 * eps : required
 * tileTokens : required
 * qOut : required
 * kOut : required
 * vOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClefGdnConvQkvVarlenGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *cuSeqlens,
    double eps,
    int64_t tileTokens,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnClefGdnConvQkvVarlen
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClefGdnConvQkvVarlen(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
